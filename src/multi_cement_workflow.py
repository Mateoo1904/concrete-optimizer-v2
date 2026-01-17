"""
multi_cement_workflow.py - FULL OPTIMIZED VERSION
✅ Không pickle predictor để tránh lỗi
✅ Support các optimization parameters mới
✅ Adaptive sizing, early stopping, caching
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
from datetime import datetime

# =========================================================================
# SAFE IMPORT
# =========================================================================
try:
    from predictor_unified import UnifiedPredictor
    from material_database import MaterialDatabase
    from nsga2_optimizer import MixDesignOptimizer
    from result_processor import ResultProcessor
    from cost_calculator import CostCalculator
    from co2_calculator import CO2Calculator
except ImportError:
    from src.predictor_unified import UnifiedPredictor
    from src.material_database import MaterialDatabase
    from src.nsga2_optimizer import MixDesignOptimizer
    from src.result_processor import ResultProcessor
    from src.cost_calculator import CostCalculator
    from src.co2_calculator import CO2Calculator
# =========================================================================


class MultiCementWorkflow:
    """
    Workflow hoàn chỉnh cho optimization nhiều loại xi măng
    ✅ OPTIMIZED: Không pickle predictor khi save results
    ✅ Support adaptive sizing, early stopping, caching
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        output_dir: str = "outputs",
        predictor: Optional[UnifiedPredictor] = None
    ):
        """
        Args:
            models_dir: Thư mục chứa trained models
            output_dir: Thư mục lưu kết quả
            predictor: UnifiedPredictor instance (nếu None thì tạo mới)
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("🔧 Initializing components...")
        
        if predictor is not None:
            print("   Using provided predictor (cached)")
            self.predictor = predictor
        else:
            print("   Creating new predictor instance")
            self.predictor = UnifiedPredictor()
        
        # ✅ Khởi tạo material_db (sẽ update density sau khi có user_input)
        self.material_db = MaterialDatabase()
        self.cost_calc = CostCalculator(self.material_db)
        self.co2_calc = CO2Calculator(self.material_db)
        self.optimizer = None
        self.processor = ResultProcessor()
        
        self.results = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("✅ Workflow initialized")
    
    def run_optimization(
        self,
        user_input: Dict,
        cement_types: List[str] = ['PC40'],
        optimization_config: Dict = None
    ) -> Dict:
        """
        Chạy optimization workflow hoàn chỉnh
        
        Args:
            user_input: Dict từ UI với yêu cầu người dùng
            cement_types: List các loại xi măng
            optimization_config: Config cho NSGA-II (bao gồm adaptive, early_stop, cache)
        
        Returns:
            {
                'optimization_results': Dict,
                'processed_results': Dict,
                'comparison': Dict,
                'recommendations': List[str]
            }
        """
        print("\n" + "="*70)
        print("🚀 MULTI-CEMENT OPTIMIZATION WORKFLOW")
        print("="*70)
        
        # Default config
        if optimization_config is None:
            optimization_config = {
                'pop_size': 100,
                'n_gen': 200,
                'seed': 42,
                'use_adaptive': True,
                'use_early_stop': True,
                'use_cache': True
            }
        
        # Step 1: Validate inputs
        print("\n📋 Step 1: Validating inputs...")
        validation = self._validate_inputs(user_input, cement_types)
        if not validation['valid']:
            print("❌ Validation failed:")
            for error in validation['errors']:
                print(f"   - {error}")
            return {'error': validation['errors']}
        print("✅ Inputs valid")
        
        # ✅ Update custom density nếu có
        if 'material_density' in user_input:
            print("   Applying custom material density...")
            self.material_db.set_custom_density(user_input['material_density'])
        
        # Step 2: Run optimization cho từng cement type
        print("\n📋 Step 2: Running optimization...")
        
        # ✅ Extract optimization parameters
        pop_size = optimization_config.get('pop_size', 100)
        n_gen = optimization_config.get('n_gen', 200)
        seed = optimization_config.get('seed', 42)
        use_adaptive = optimization_config.get('use_adaptive', True)
        use_early_stop = optimization_config.get('use_early_stop', True)
        use_cache = optimization_config.get('use_cache', True)
        
        self.optimizer = MixDesignOptimizer(
            predictor=self.predictor,
            material_db=self.material_db,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            use_adaptive=use_adaptive,
            use_early_stop=use_early_stop
        )
        
        optimization_results = self.optimizer.optimize(
            user_input=user_input,
            cement_types=cement_types,
            verbose=True
        )
        
        # Step 3: Process results
        print("\n📋 Step 3: Processing results...")
        processed_results = self.processor.process_results(
            optimization_results,
            user_preferences=user_input.get('preferences', None)
        )
        
        # Step 4: Compare cement types (nếu có nhiều)
        comparison = {}
        if len(cement_types) > 1:
            print("\n📋 Step 4: Comparing cement types...")
            comparison = self._compare_cement_types(processed_results)
        
        # Step 5: Generate recommendations
        print("\n📋 Step 5: Generating recommendations...")
        recommendations = self._generate_recommendations(
            processed_results,
            comparison,
            user_input
        )
        
        # Save results
        self.results = {
            'optimization_results': optimization_results,
            'processed_results': processed_results,
            'comparison': comparison,
            'recommendations': recommendations,
            'user_input': user_input,
            'session_id': self.session_id
        }
        
        # ✅ FIX: Chỉ save CSV, KHÔNG pickle toàn bộ results
        self._save_results_csv_only()
        
        print("\n" + "="*70)
        print("✅ WORKFLOW COMPLETE")
        print("="*70)
        
        return self.results
    
    def _validate_inputs(
        self,
        user_input: Dict,
        cement_types: List[str]
    ) -> Dict:
        """Validate user inputs"""
        errors = []
        
        # Check required fields
        required = ['fc_target', 'age_target', 'slump_target', 'slump_tolerance']
        for field in required:
            if field not in user_input:
                errors.append(f"Missing required field: {field}")
        
        # Check ranges
        if 'fc_target' in user_input:
            if not (15 <= user_input['fc_target'] <= 80):
                errors.append("fc_target must be between 15-80 MPa")
        
        if 'slump_target' in user_input:
            if not (50 <= user_input['slump_target'] <= 250):
                errors.append("slump_target must be between 50-250 mm")
        
        # Check cement types
        valid_cements = ['PC40', 'PC50']
        for ct in cement_types:
            if ct not in valid_cements:
                errors.append(f"Invalid cement type: {ct}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _compare_cement_types(self, processed_results: Dict) -> Dict:
        """So sánh giữa các loại xi măng"""
        if len(processed_results) < 2:
            return {}
        
        cement_types = list(processed_results.keys())
        comparisons = {}
        
        for i in range(len(cement_types)):
            for j in range(i+1, len(cement_types)):
                ct1, ct2 = cement_types[i], cement_types[j]
                
                design1 = processed_results[ct1]['ranked_designs'][0]
                design2 = processed_results[ct2]['ranked_designs'][0]
                
                cost1 = design1['objectives']['cost']
                cost2 = design2['objectives']['cost']
                
                strength1 = design1['predictions']['f28']
                strength2 = design2['predictions']['f28']
                
                co2_1 = design1['objectives']['co2']
                co2_2 = design2['objectives']['co2']
                
                key = f"{ct1}_vs_{ct2}"
                comparisons[key] = {
                    'cost_difference': cost2 - cost1,
                    'cost_pct': (cost2 - cost1) / cost1 * 100,
                    'strength_difference': strength2 - strength1,
                    'strength_pct': (strength2 - strength1) / strength1 * 100,
                    'co2_difference': co2_2 - co2_1,
                    'co2_pct': (co2_2 - co2_1) / co2_1 * 100,
                    'recommendation': self._make_recommendation(cost1, cost2, strength1, strength2)
                }
        
        return comparisons
    
    def _make_recommendation(self, cost1, cost2, str1, str2):
        """Tạo recommendation dựa trên so sánh"""
        cost_diff_pct = (cost2 - cost1) / cost1 * 100
        str_diff_pct = (str2 - str1) / str1 * 100
        
        if cost_diff_pct < 5 and str_diff_pct > 10:
            return "Cement type 2 tốt hơn (chi phí tương đương, mạnh hơn nhiều)"
        elif cost_diff_pct > 10 and str_diff_pct < 5:
            return "Cement type 1 tốt hơn (rẻ hơn nhiều, cường độ tương đương)"
        else:
            return "Cả hai đều khả thi, tùy thuộc ưu tiên"
    
    def _generate_recommendations(
        self,
        processed_results: Dict,
        comparison: Dict,
        user_input: Dict
    ) -> List[str]:
        """Generate recommendations dựa trên kết quả"""
        recommendations = []
        
        for cement_type, result in processed_results.items():
            top_design = result['ranked_designs'][0]
            
            # Recommendation 1: Cost efficiency
            cost = top_design['objectives']['cost']
            if cost < 120000:
                recommendations.append(
                    f"✅ {cement_type}: Chi phí thấp ({cost:,.0f} VNĐ/m³) - Phù hợp cho dự án kinh tế"
                )
            elif cost > 150000:
                recommendations.append(
                    f"⚠️ {cement_type}: Chi phí cao ({cost:,.0f} VNĐ/m³) - Cân nhắc giảm SCM hoặc thay xi măng"
                )
            
            # Recommendation 2: Strength
            strength = top_design['predictions']['f28']
            target = user_input['fc_target']
            if strength >= target * 1.1:
                recommendations.append(
                    f"✅ {cement_type}: Cường độ vượt mức {(strength-target)/target*100:.0f}% - Có thể giảm binder để tiết kiệm"
                )
            elif strength < target:
                recommendations.append(
                    f"❌ {cement_type}: Cường độ không đạt - Cần tăng binder hoặc giảm w/b"
                )
            
            # Recommendation 3: Sustainability
            co2 = top_design['objectives']['co2']
            if co2 < 300:
                recommendations.append(
                    f"🌱 {cement_type}: Phát thải thấp ({co2:.0f} kgCO2/m³) - Thân thiện môi trường"
                )
            elif co2 > 400:
                recommendations.append(
                    f"⚠️ {cement_type}: Phát thải cao ({co2:.0f} kgCO2/m³) - Cân nhắc tăng SCM"
                )
            
            # Recommendation 4: Workability
            slump_dev = top_design['objectives']['slump_deviation']
            if slump_dev < 10:
                recommendations.append(
                    f"✅ {cement_type}: Độ sụt chính xác (sai lệch {slump_dev:.0f} mm)"
                )
            elif slump_dev > 20:
                recommendations.append(
                    f"⚠️ {cement_type}: Độ sụt sai lệch lớn ({slump_dev:.0f} mm) - Cần điều chỉnh SP"
                )
        
        # Comparison recommendations
        if comparison:
            recommendations.append("\n📊 SO SÁNH:")
            for key, comp in comparison.items():
                recommendations.append(f"   {key}: {comp['recommendation']}")
        
        return recommendations
    
    def _save_results_csv_only(self):
        """
        ✅ FIX: Chỉ lưu CSV (Pareto front + recommendations)
        KHÔNG pickle toàn bộ results vì chứa predictor
        """
        session_dir = self.output_dir / self.session_id
        session_dir.mkdir(exist_ok=True)
        
        try:
            # Save Pareto fronts as CSV
            for cement_type, opt_result in self.results['optimization_results'].items():
                X, F = opt_result['pareto_front']
                
                df = pd.DataFrame(X, columns=[
                    'cement', 'water', 'flyash', 'slag', 'silica_fume',
                    'superplasticizer', 'fine_agg', 'coarse_agg'
                ])
                df['cost'] = F[:, 0]
                df['strength'] = -F[:, 1]
                df['slump_deviation'] = F[:, 2]
                df['co2'] = F[:, 3]
                
                df.to_csv(session_dir / f"pareto_front_{cement_type}.csv", index=False)
            
            # Save recommendations
            with open(session_dir / "recommendations.txt", 'w', encoding='utf-8') as f:
                f.write("RECOMMENDATIONS\n")
                f.write("="*70 + "\n\n")
                for rec in self.results['recommendations']:
                    f.write(rec + "\n")
            
            print(f"\n💾 Results saved to: {session_dir}")
            print("   ✅ Pareto fronts (CSV)")
            print("   ✅ Recommendations (TXT)")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save some results: {e}")
            # Không crash app, chỉ warning
    
    def export_for_production(
        self,
        design_ids: List[Tuple[str, int]]
    ) -> str:
        """
        Export selected designs cho production
        
        Args:
            design_ids: List of (cement_type, design_index)
        
        Returns:
            Path to export file
        """
        export_data = []
        
        for cement_type, idx in design_ids:
            design = self.results['processed_results'][cement_type]['ranked_designs'][idx]
            
            mix = design['mix_design']
            
            export_data.append({
                'cement_type': cement_type,
                'profile': design['profile'],
                
                # Mix proportions
                'cement_kg': mix['cement'],
                'water_kg': mix['water'],
                'flyash_kg': mix.get('flyash', 0),
                'slag_kg': mix.get('slag', 0),
                'silica_fume_kg': mix.get('silica_fume', 0),
                'sp_kg': mix.get('superplasticizer', 0),
                'fine_agg_kg': mix['fine_agg'],
                'coarse_agg_kg': mix['coarse_agg'],
                
                # Properties
                'w_b': mix['water'] / (mix['cement'] + mix.get('flyash', 0) + 
                                       mix.get('slag', 0) + mix.get('silica_fume', 0)),
                
                # Predictions
                'predicted_f28_MPa': design['predictions']['f28'],
                'predicted_slump_mm': design['predictions']['slump'],
                
                # Economics
                'cost_VND_per_m3': design['objectives']['cost'],
                'co2_kg_per_m3': design['objectives']['co2']
            })
        
        df = pd.DataFrame(export_data)
        
        export_path = self.output_dir / self.session_id / "production_designs.csv"
        df.to_csv(export_path, index=False)
        
        print(f"📤 Production designs exported to: {export_path}")
        
        return str(export_path)


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    workflow = MultiCementWorkflow(
        models_dir="models",
        output_dir="outputs"
    )
    
    user_input = {
        'fc_target': 40.0,
        'age_target': 28,
        'slump_target': 180,
        'slump_tolerance': 20,
        'available_materials': {
            'Xỉ (Slag)': {'available': True, 'category': 'SCM'},
            'Tro bay (Flyash)': {'available': True, 'category': 'SCM'},
            'Silica fume': {'available': False},
            'Phụ gia siêu dẻo (SP)': {'available': True}
        },
        'preferences': {
            'cost': 0.4,
            'performance': 0.3,
            'sustainability': 0.2,
            'workability': 0.1
        }
    }
    
    # Run workflow
    results = workflow.run_optimization(
        user_input=user_input,
        cement_types=['PC40', 'PC50'],
        optimization_config={
            'pop_size': 100, 
            'n_gen': 200,
            'use_adaptive': True,
            'use_early_stop': True,
            'use_cache': True
        }
    )
    
    # Export cho production
    export_path = workflow.export_for_production([
        ('PC40', 0),
        ('PC50', 0)
    ])
    
    print("\n✅ Workflow complete!")