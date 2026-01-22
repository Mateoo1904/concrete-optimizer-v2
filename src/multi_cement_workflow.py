"""
multi_cement_workflow.py - FIXED VERSION
✅ Handle empty ranked_designs gracefully
✅ Better validation and error messages
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
from datetime import datetime
import traceback

# SAFE IMPORT
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


class MultiCementWorkflow:
    """
    Workflow hoàn chỉnh cho optimization nhiều loại xi măng
    ✅ FIXED: Handle empty results gracefully
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        output_dir: str = "outputs",
        predictor: Optional[UnifiedPredictor] = None
    ):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔧 Initializing components...")
        
        if predictor is not None:
            print("   Using provided predictor (cached)")
            self.predictor = predictor
        else:
            print("   Creating new predictor instance")
            self.predictor = UnifiedPredictor()
        
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
        ✅ FIXED: Handle empty results
        """
        print("\n" + "="*70)
        print("🚀 MULTI-CEMENT OPTIMIZATION WORKFLOW")
        print("="*70)
        
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
        
        # Update custom density if provided
        if 'material_density' in user_input:
            print("   Applying custom material density...")
            self.material_db.set_custom_density(user_input['material_density'])
        
        # Step 2: Run optimization
        print("\n📋 Step 2: Running optimization...")
        
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
        
        try:
            optimization_results = self.optimizer.optimize(
                user_input=user_input,
                cement_types=cement_types,
                verbose=True
            )
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            traceback.print_exc()
            return {
                'error': f"Optimization failed: {str(e)}",
                'optimization_results': {},
                'processed_results': {},
                'comparison': {},
                'recommendations': [f"❌ Optimization failed: {str(e)}"]
            }
        
        # Step 3: Process results
        print("\n📋 Step 3: Processing results...")
        try:
            processed_results = self.processor.process_results(
                optimization_results,
                user_preferences=user_input.get('preferences', None)
            )
        except Exception as e:
            print(f"❌ Processing error: {e}")
            traceback.print_exc()
            processed_results = {}
        
        # Step 4: Compare cement types (if multiple)
        comparison = {}
        if len(cement_types) > 1:
            print("\n📋 Step 4: Comparing cement types...")
            try:
                comparison = self._compare_cement_types(processed_results)
            except Exception as e:
                print(f"⚠️ Comparison error: {e}")
                comparison = {}
        
        # Step 5: Generate recommendations
        print("\n📋 Step 5: Generating recommendations...")
        try:
            recommendations = self._generate_recommendations(
                processed_results,
                comparison,
                user_input
            )
        except Exception as e:
            print(f"⚠️ Recommendation error: {e}")
            recommendations = [f"⚠️ Could not generate recommendations: {str(e)}"]
        
        # Save results
        self.results = {
            'optimization_results': optimization_results,
            'processed_results': processed_results,
            'comparison': comparison,
            'recommendations': recommendations,
            'user_input': user_input,
            'session_id': self.session_id
        }
        
        self._save_results_csv_only()
        
        print("\n" + "="*70)
        print("✅ WORKFLOW COMPLETE")
        print("="*70)
        
        return self.results
    
    def _validate_inputs(self, user_input: Dict, cement_types: List[str]) -> Dict:
        """Validate user inputs"""
        errors = []
        
        required = ['fc_target', 'age_target', 'slump_target', 'slump_tolerance']
        for field in required:
            if field not in user_input:
                errors.append(f"Missing required field: {field}")
        
        if 'fc_target' in user_input:
            if not (15 <= user_input['fc_target'] <= 80):
                errors.append("fc_target must be between 15-80 MPa")
        
        if 'slump_target' in user_input:
            if not (50 <= user_input['slump_target'] <= 250):
                errors.append("slump_target must be between 50-250 mm")
        
        valid_cements = ['PC40', 'PC50']
        for ct in cement_types:
            if ct not in valid_cements:
                errors.append(f"Invalid cement type: {ct}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _compare_cement_types(self, processed_results: Dict) -> Dict:
        """So sánh giữa các loại xi măng - ✅ FIXED"""
        if len(processed_results) < 2:
            return {}
        
        cement_types = list(processed_results.keys())
        comparisons = {}
        
        for i in range(len(cement_types)):
            for j in range(i+1, len(cement_types)):
                ct1, ct2 = cement_types[i], cement_types[j]
                
                # ✅ CHECK: Ensure ranked_designs exist
                if not processed_results[ct1]['ranked_designs'] or not processed_results[ct2]['ranked_designs']:
                    print(f"⚠️ Cannot compare {ct1} vs {ct2}: missing designs")
                    continue
                
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
                    'cost_pct': (cost2 - cost1) / cost1 * 100 if cost1 > 0 else 0,
                    'strength_difference': strength2 - strength1,
                    'strength_pct': (strength2 - strength1) / strength1 * 100 if strength1 > 0 else 0,
                    'co2_difference': co2_2 - co2_1,
                    'co2_pct': (co2_2 - co2_1) / co2_1 * 100 if co2_1 > 0 else 0,
                    'recommendation': self._make_recommendation(cost1, cost2, strength1, strength2)
                }
        
        return comparisons
    
    def _make_recommendation(self, cost1, cost2, str1, str2):
        """Tạo recommendation dựa trên so sánh"""
        if cost1 == 0 or str1 == 0:
            return "Insufficient data for comparison"
        
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
        """
        Generate recommendations - ✅ FIXED: Handle empty designs
        """
        recommendations = []
        
        for cement_type, result in processed_results.items():
            # ✅ CHECK: Ensure ranked_designs exist
            if not result['ranked_designs']:
                recommendations.append(
                    f"⚠️ {cement_type}: No valid designs found. Try relaxing constraints or increasing population/generations."
                )
                continue
            
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
        """Save CSV results only"""
        session_dir = self.output_dir / self.session_id
        session_dir.mkdir(exist_ok=True)
        
        try:
            for cement_type, opt_result in self.results['optimization_results'].items():
                if 'error' in opt_result:
                    continue
                
                X, F = opt_result['pareto_front']
                
                if len(X) == 0:
                    print(f"⚠️ No solutions to save for {cement_type}")
                    continue
                
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
    
    def export_for_production(self, design_ids: List[Tuple[str, int]]) -> str:
        """Export selected designs for production"""
        export_data = []
        
        for cement_type, idx in design_ids:
            if cement_type not in self.results['processed_results']:
                continue
            
            ranked = self.results['processed_results'][cement_type]['ranked_designs']
            
            if idx >= len(ranked):
                print(f"⚠️ Design index {idx} out of range for {cement_type}")
                continue
            
            design = ranked[idx]
            mix = design['mix_design']
            
            export_data.append({
                'cement_type': cement_type,
                'profile': design['profile'],
                'cement_kg': mix['cement'],
                'water_kg': mix['water'],
                'flyash_kg': mix.get('flyash', 0),
                'slag_kg': mix.get('slag', 0),
                'silica_fume_kg': mix.get('silica_fume', 0),
                'sp_kg': mix.get('superplasticizer', 0),
                'fine_agg_kg': mix['fine_agg'],
                'coarse_agg_kg': mix['coarse_agg'],
                'w_b': mix['water'] / (mix['cement'] + mix.get('flyash', 0) + 
                                       mix.get('slag', 0) + mix.get('silica_fume', 0)),
                'predicted_f28_MPa': design['predictions']['f28'],
                'predicted_slump_mm': design['predictions']['slump'],
                'cost_VND_per_m3': design['objectives']['cost'],
                'co2_kg_per_m3': design['objectives']['co2']
            })
        
        if not export_data:
            print("⚠️ No valid designs to export")
            return ""
        
        df = pd.DataFrame(export_data)
        export_path = self.output_dir / self.session_id / "production_designs.csv"
        df.to_csv(export_path, index=False)
        
        print(f"📤 Production designs exported to: {export_path}")
        return str(export_path)


if __name__ == "__main__":
    print("✅ multi_cement_workflow.py - WITH ROBUST ERROR HANDLING!")
