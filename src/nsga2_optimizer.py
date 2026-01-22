"""
nsga2_optimizer.py - FIXED VERSION WITH PROPER ERROR HANDLING
✅ Kiểm tra None trước khi access
✅ Better error messages
✅ Graceful degradation
"""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from typing import Dict, List
import time
import traceback

# SAFE IMPORT
try:
    from optimization_problem import ConcreteMixOptimizationProblem
    from predictor_unified import UnifiedPredictor
    from constraint_builder import ConstraintBuilder
    from material_database import MaterialDatabase
except ImportError:
    from src.optimization_problem import ConcreteMixOptimizationProblem
    from src.predictor_unified import UnifiedPredictor
    from src.constraint_builder import ConstraintBuilder
    from src.material_database import MaterialDatabase


class MixDesignOptimizer:
    """
    NSGA-II optimizer - WITH ROBUST ERROR HANDLING
    """

    def __init__(
        self,
        predictor: UnifiedPredictor,
        material_db: MaterialDatabase,
        pop_size: int = 100,
        n_gen: int = 200,
        seed: int = 42,
        use_adaptive: bool = True,
        use_early_stop: bool = True
    ):
        self.predictor = predictor
        self.material_db = material_db
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.use_adaptive = use_adaptive
        self.use_early_stop = use_early_stop
        self.results = {}

    def optimize(
        self,
        user_input: Dict,
        cement_types: List[str] = ['PC40'],
        verbose: bool = True
    ) -> Dict:
        """
        Chạy optimization cho các loại xi măng
        """
        results_all = {}

        for cement_type in cement_types:
            if verbose:
                print(f"\n{'='*70}")
                print(f"🔄 Optimizing for {cement_type}")
                print(f"{'='*70}")

            try:
                result = self._optimize_single_cement(
                    user_input, cement_type, verbose
                )
                results_all[cement_type] = result
            except Exception as e:
                print(f"❌ Error optimizing {cement_type}: {e}")
                traceback.print_exc()
                # Return empty result instead of crashing
                results_all[cement_type] = {
                    'pareto_front': (np.array([]), np.array([])),
                    'top_designs': [],
                    'metrics': {
                        'n_solutions': 0,
                        'cost_range': (0, 0),
                        'strength_range': (0, 0),
                        'co2_range': (0, 0),
                        'avg_slump_deviation': 0
                    },
                    'problem': None,
                    'optimization_time': 0,
                    'error': str(e)
                }

        self.results = results_all
        return results_all

    def _optimize_single_cement(
        self,
        user_input: Dict,
        cement_type: str,
        verbose: bool
    ) -> Dict:
        """Optimize cho 1 loại xi măng - WITH ERROR HANDLING"""

        try:
            # Build constraints
            builder = ConstraintBuilder(self.material_db)
            constraint_config = builder.build_from_user_input(user_input)

            if verbose:
                print(builder.get_constraint_summary())

            # Adaptive population size
            if self.use_adaptive:
                n_active_scm = sum([
                    1 for k in ['flyash', 'slag', 'silica_fume'] 
                    if constraint_config['bounds'][k][1] > 0
                ])
                
                if n_active_scm == 0:
                    actual_pop_size = max(int(self.pop_size * 0.7), 50)
                    actual_n_gen = max(int(self.n_gen * 0.7), 100)
                elif n_active_scm == 1:
                    actual_pop_size = max(int(self.pop_size * 0.85), 70)
                    actual_n_gen = max(int(self.n_gen * 0.85), 150)
                else:
                    actual_pop_size = self.pop_size
                    actual_n_gen = self.n_gen
                
                if verbose:
                    print(f"\n📊 Adaptive sizing: pop={actual_pop_size}, gen={actual_n_gen}")
            else:
                actual_pop_size = self.pop_size
                actual_n_gen = self.n_gen

            # Create problem
            problem = ConcreteMixOptimizationProblem(
                self.predictor,
                constraint_config,
                cement_type
            )

            # Setup algorithm
            algorithm = NSGA2(
                pop_size=actual_pop_size,
                sampling=LatinHypercubeSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=0.2, eta=20),
                eliminate_duplicates=True
            )

            # Termination
            termination = get_termination("n_gen", actual_n_gen)

            # Run optimization
            if verbose:
                print(f"\n🚀 Running NSGA-II...")
                print(f"   Population: {actual_pop_size}")
                print(f"   Generations: {actual_n_gen}")
                
            start_time = time.time()

            res = minimize(
                problem,
                algorithm,
                termination=termination,
                seed=self.seed,
                verbose=verbose,
                save_history=False
            )

            elapsed = time.time() - start_time

            # ✅ FIX: Kiểm tra res.X và res.F trước khi sử dụng
            if res.X is None or res.F is None:
                print(f"❌ res.X is None: {res.X is None}, res.F is None: {res.F is None}")
                raise ValueError(f"Optimization returned None for {cement_type}")
            
            if len(res.X) == 0:
                print(f"❌ Optimization found 0 solutions for {cement_type}")
                print(f"   This usually means constraints are too strict")
                print(f"   Try: 1) Relaxing constraints, 2) Increasing pop_size/n_gen, 3) Checking bounds")
                raise ValueError(f"Optimization found no valid solutions for {cement_type}")

            X_pareto = res.X
            F_pareto = res.F

            if verbose:
                print(f"\n✅ Optimization complete!")
                print(f"   Time: {elapsed:.1f}s")
                print(f"   Pareto front: {len(X_pareto)} solutions")
                print(f"   Speed: {len(X_pareto)/elapsed:.1f} solutions/sec")

            # Select diverse designs
            top_designs = self._select_diverse_designs(
                X_pareto, F_pareto, problem, user_input, n=5
            )

            # Calculate metrics
            metrics = self._calculate_metrics(F_pareto)

            return {
                'pareto_front': (X_pareto, F_pareto),
                'top_designs': top_designs,
                'metrics': metrics,
                'problem': problem,
                'optimization_time': elapsed
            }

        except Exception as e:
            print(f"❌ Error in _optimize_single_cement: {e}")
            traceback.print_exc()
            raise  # Re-raise để outer function xử lý

    def _select_diverse_designs(
        self,
        X: np.ndarray,
        F: np.ndarray,
        problem: ConcreteMixOptimizationProblem,
        user_input: Dict,
        n: int = 5
    ) -> List[Dict]:
        """
        ✅ FIXED: Kiểm tra empty arrays + Better logging
        """
        print(f"\n🔍 _select_diverse_designs called:")
        print(f"   X shape: {X.shape}")
        print(f"   F shape: {F.shape}")
        
        if len(X) == 0 or len(F) == 0:
            print("❌ No solutions to select from (X or F is empty)")
            return []

        designs = []
        selected_indices = set()
        fc_target = user_input.get('fc_target', 40)
        
        print(f"   fc_target: {fc_target}")
        
        # Tính actual strength cho mỗi design
        actual_strengths = []
        for i in range(len(X)):
            try:
                mix = problem._x_to_mix_dict(X[i])
                predictions = problem.predictor.predict_all(mix)
                predictions = problem.adjuster.adjust_predictions(
                    mix, predictions, problem.cement_type
                )
                
                age_target = user_input['age_target']
                if age_target == 28:
                    fc_age = predictions['f28']
                else:
                    fc_age = problem.predictor.predict_strength_at_age(mix, age_target)
                    fc_age = problem.adjuster.adjust_predictions(
                        mix, {'f28': fc_age}, problem.cement_type
                    )['f28']
                
                actual_strengths.append(fc_age)
            except Exception as e:
                print(f"   ⚠️ Error computing strength for design {i}: {e}")
                actual_strengths.append(0)
        
        actual_strengths = np.array(actual_strengths)
        print(f"   Actual strengths range: [{actual_strengths.min():.1f}, {actual_strengths.max():.1f}] MPa")
        print(f"   Designs meeting fc_target: {np.sum(actual_strengths >= fc_target)}/{len(actual_strengths)}")

        # 1. Cost Optimized
        idx_cheap = np.argmin(F[:, 0])
        selected_indices.add(idx_cheap)
        designs.append(self._format_design(X[idx_cheap], F[idx_cheap], problem, "Cost Optimized"))

        # 2. Strength Optimized
        target_min = fc_target * 1.25
        target_max = fc_target * 1.30
        target_strength = (target_min + target_max) / 2
        
        valid_mask = actual_strengths >= fc_target
        available_mask = np.ones(len(X), dtype=bool)
        for idx in selected_indices:
            available_mask[idx] = False
        
        combined_mask = valid_mask & available_mask
        
        if np.any(combined_mask):
            distances = np.abs(actual_strengths - target_strength)
            filtered_distances = np.where(combined_mask, distances, np.inf)
            idx_strong = np.argmin(filtered_distances)
        else:
            filtered_strengths = np.where(available_mask, actual_strengths, -np.inf)
            idx_strong = np.argmax(filtered_strengths)
        
        selected_indices.add(idx_strong)
        designs.append(self._format_design(X[idx_strong], F[idx_strong], problem, "Strength Optimized"))

        # 3. Eco-friendly
        available_mask = np.ones(len(X), dtype=bool)
        for idx in selected_indices:
            available_mask[idx] = False
        
        co2_filtered = np.where(available_mask, F[:, 3], np.inf)
        idx_eco = np.argmin(co2_filtered)
        selected_indices.add(idx_eco)
        designs.append(self._format_design(X[idx_eco], F[idx_eco], problem, "Eco-friendly"))

        # 4. Balanced
        available_mask = np.ones(len(X), dtype=bool)
        for idx in selected_indices:
            available_mask[idx] = False
        
        available_indices = np.where(available_mask)[0]
        if len(available_indices) > 0:
            F_available = F[available_indices]
            F_norm = (F_available - F_available.min(axis=0)) / (F_available.max(axis=0) - F_available.min(axis=0) + 1e-10)
            distances = np.sqrt(np.sum(F_norm**2, axis=1))
            idx_knee_local = np.argmin(distances)
            idx_knee = available_indices[idx_knee_local]
        else:
            idx_knee = 0
        
        selected_indices.add(idx_knee)
        designs.append(self._format_design(X[idx_knee], F[idx_knee], problem, "Balanced"))

        # 5. Slump Optimized
        available_mask = np.ones(len(X), dtype=bool)
        for idx in selected_indices:
            available_mask[idx] = False
        
        slump_dev_filtered = np.where(available_mask, F[:, 2], np.inf)
        idx_slump = np.argmin(slump_dev_filtered)
        selected_indices.add(idx_slump)
        designs.append(self._format_design(X[idx_slump], F[idx_slump], problem, "Slump Optimized"))

        print(f"   ✅ Selected {len(designs)} diverse designs")
        for i, d in enumerate(designs, 1):
            print(f"      {i}. {d['profile']}: f28={d['predictions']['f28']:.1f} MPa, cost={d['objectives']['cost']:,.0f} VNĐ")
        
        return designs

    def _format_design(
        self,
        x: np.ndarray,
        f: np.ndarray,
        problem: ConcreteMixOptimizationProblem,
        profile: str
    ) -> Dict:
        """Format design - WITH ERROR HANDLING"""
        try:
            mix = problem._x_to_mix_dict(x)

            # Predictions
            predictions = problem.predictor.predict_all(mix)
            predictions = problem.adjuster.adjust_predictions(
                mix, predictions, problem.cement_type
            )

            # Cost & CO2
            cost_data = problem.cost_calc.calculate_total_cost(mix, problem.cement_type)
            co2_data = problem.co2_calc.calculate_total_emission(mix, problem.cement_type)

            # Validation
            is_valid, violations = problem.validate_mix(mix)

            # Score
            f_norm = f.copy()
            f_norm[1] = -f_norm[1]
            
            f_scaled = np.array([
                f_norm[0] / 1000000,
                f_norm[1] / 50,
                f_norm[2] / 50,
                f_norm[3] / 500
            ])
            score = 1.0 / (1.0 + np.linalg.norm(f_scaled))

            return {
                'profile': profile,
                'cement_type': problem.cement_type,
                'mix_design': mix,
                'predictions': {
                    'f28': predictions['f28'],
                    's': predictions['s'],
                    'slump': predictions['slump']
                },
                'objectives': {
                    'cost': f[0],
                    'strength': predictions['f28'],
                    'slump_deviation': f[2],
                    'co2': f[3]
                },
                'cost_breakdown': cost_data['breakdown'],
                'co2_breakdown': co2_data['breakdown'],
                'validation': {
                    'is_valid': is_valid,
                    'violations': violations
                },
                'score': score
            }
        except Exception as e:
            print(f"⚠️ Error formatting design: {e}")
            # Return minimal valid design
            return {
                'profile': profile,
                'cement_type': problem.cement_type,
                'mix_design': {},
                'predictions': {'f28': 0, 's': 0.25, 'slump': 0},
                'objectives': {'cost': 0, 'strength': 0, 'slump_deviation': 0, 'co2': 0},
                'cost_breakdown': {},
                'co2_breakdown': {},
                'validation': {'is_valid': False, 'violations': ['Error formatting design']},
                'score': 0
            }

    def _calculate_metrics(self, F: np.ndarray) -> Dict:
        """Tính metrics - WITH SAFETY CHECKS"""
        if len(F) == 0:
            return {
                'n_solutions': 0,
                'cost_range': (0, 0),
                'strength_range': (0, 0),
                'co2_range': (0, 0),
                'avg_slump_deviation': 0
            }
        
        return {
            'n_solutions': len(F),
            'cost_range': (float(F[:, 0].min()), float(F[:, 0].max())),
            'strength_range': (float(-F[:, 1].max()), float(-F[:, 1].min())),
            'co2_range': (float(F[:, 3].min()), float(F[:, 3].max())),
            'avg_slump_deviation': float(F[:, 2].mean())
        }


# ===== TEST =====
if __name__ == "__main__":
    print("✅ nsga2_optimizer.py - WITH ROBUST ERROR HANDLING!")
