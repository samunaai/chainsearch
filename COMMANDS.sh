# =============================================================================
# EXPERIMENT FLOWCHART 😊
# =============================================================================
#
# 1. pretraining
#          |
# 2. heuristic fine-tuning
#          |
# 3. held-out generations + labels + cached activations
#          |
# 4. compute steering vectors
#          |
#          +---------------------------------------------+
#          |                      |                      |
# linear-probe vectors       CAA vectors       direct optimisation
#                                                 vectors
#          |                      |                      |
#          +----------------------+----------------------+
#                                 |
#              5. vector comparison + steering evaluation
#
# Probe and CAA vectors are estimated from cached activations.
# Direct optimisation uses live forward and backward passes through the
# frozen model while updating only the local steering vector.
#

# =============================================================================
# 1) PRETRAINING (SETTING ASIDE FINE-TUNING SAMPLE INDICES FOR LATER)
# =============================================================================

# Generate Colton Dataset using HR3 system
python generate_colton_hr3_dataset.py --workers 96 &> out_colton.txt

# [JUPYTER] Use notebooks_for_inspection/inspect_astar.ipynb
# to inspect generated A* traces and sample examples by BACK count.

# Generate A* dataset
# A* is not used as pretraining data. We generate it early only so
# generate_pretrain_sample_list.py can remove exact (inputs, target)
# overlaps from the HR3 pretraining source.
python generate_astar_dataset.py &> out_astar.txt

# [JUPYTER] Use notebooks_for_inspection/inspect_astar.ipynb
# to inspect generated A* traces and sample examples by BACK count.

# Generate HR3 \ A* source records for pretraining
python generate_pretrain_sample_list.py &> out_setminus.txt

# Check HR3 \ A* really has no exact (inputs, target) overlap with A*
python quick_check_hr3_astar_no_overlap.py &> out_hr3_astar_overlap_check.txt

# Generate final pretraining traces by adding random backtracks to HR3 \ A*
python generate_final_pretrain_traces.py &> out_pretrain_traces.txt

# [JUPYTER] Use notebooks_for_inspection/inspect_final_pretrain_traces.ipynb
# to verify the final pretraining traces and review representative examples.

# Pretrain model with HR3-derived randomised pretraining traces
COUNTDOWN_STAGE=pretrain CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 pretrain_and_sft_model.py \
&> out_pretrain_model.txt

# [JUPYTER] Use notebooks_for_inspection/inspect_model_pretrain_finetune.ipynb
# to visually compare pretrained-model generations with validation targets.

# =============================================================================
# 2) HEURISTIC FINETUNING (MODIFIED A*)
# =============================================================================

# Split A* dataset into target-disjoint train/val/test/probe sets
python -u generate_custom_astar_splits.py \
  > out_custom_astar_splits.txt 2>&1  

# Check the A* train/val/test/probe split for index and target overlap
python -u quick_check_astar_split_no_overlap.py \
  > out_astar_split_overlap_check.txt 2>&1

# Fine-tune model with A* data
# Important: trainer stage key is "astar", even though internally it is fine-tuning
COUNTDOWN_STAGE=astar CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 pretrain_and_sft_model.py \
&> out_finetune_model.txt
 
# =============================================================================
# 3) HELDOUT SAMPLE AND LABEL GENERATION, PLUS ACTIVATION CACHING
# =============================================================================

# Use the fine-tuned A* model on the held-out probing prompts,
# evaluate final answers and trace validity, and save all generated traces
CUDA_VISIBLE_DEVICES=0 python -u evaluate_unused_astar_samples_for_probing.py \
&> out_evaluate_unused_astar_samples_for_probing.txt

# Check every generated trace and label whether each valid state can still
# reach the target without `BACK`
python -u label_generated_states_for_probing_samples.py \
  --workers 24 \
  --no-resume \
&> out_label_generated_states_for_probing_samples.txt

# [Optional] Reachability diagnostic
# Measure how forward-unreachable states vary with trace BACK count.
python -u analyze_astar_reachability_by_backtrack_count.py \
&> out_analyze_astar_reachability_by_backtrack_count.txt 

# Cache activations used by the linear probes and CAA
CUDA_VISIBLE_DEVICES=0 python -u build_probe_dataset.py --batch-size 32 \
  > out_build_probe_dataset.txt 2>&1
status=$?
echo "" >> out_build_probe_dataset.txt
echo "EXIT STATUS: $status" >> out_build_probe_dataset.txt

# [Optional] Verify the structure of BACK transitions
python -u verify_no_reachable_to_unreachable_backtracks.py \
  &> out_verify_no_reachable_to_unreachable_backtracks.txt

# =============================================================================
# 4) THREE VECTOR TYPES (PROBE, CAA, AND DIRECT OPTIMISATION)
# =============================================================================

# a) Train linear probes on the cached A* activations and save their weight vectors
# previously called `compute_linear_probe_vectors` as `train_and_evaluate_astar_probes.py`
# so some old experiments are written to `out_train_and_evaluate_astar_probes.txt`
CUDA_VISIBLE_DEVICES=0 python -u compute_linear_probe_vectors.py \
  --workers 24 \
  > out_compute_linear_probe_vectors.txt 2>&1
status=$?
echo "" >> out_compute_linear_probe_vectors.txt
echo "EXIT STATUS: $status" >> out_compute_linear_probe_vectors.txt

# Plot probe performance. Use `--plotlist` to select a subset
python -u plot_probe_results.py --plotlist all --output-dir exp34_probe_plots &> out_plot_probe_results.txt
# Faster main-figure-only rerun:
# python -u plot_probe_results.py --plotlist decision_point_progress --output-dir exp34_probe_plots

# b) Compute CAA vectors
CUDA_VISIBLE_DEVICES=0 python -u compute_caa_vectors.py \
  > out_compute_caa_vectors.txt 2>&1
status=$?
echo "" >> out_compute_caa_vectors.txt
echo "EXIT STATUS: $status" >> out_compute_caa_vectors.txt

# c) Compute direct optimisation vectors
CUDA_VISIBLE_DEVICES=0 python -u compute_direct_optimisation_vectors.py \
  --no-resume --overwrite \
  > out_compute_direct_optimisation_vectors.txt 2>&1
status=$?
echo "" >> out_compute_direct_optimisation_vectors.txt
echo "EXIT STATUS: $status" >> out_compute_direct_optimisation_vectors.txt
# can monitor progress, as always, with a command like 
# tail -f out_compute_direct_optimisation_vectors.txt

# Check that Direct Optimisation converged sufficiently
python -u analyze_direct_optimisation_convergence.py \
  > out_analyze_direct_optimisation_convergence.txt 2>&1

# =============================================================================
# 5) ANALYSIS: STEERING AND COMPARATIVE EVALUATION
# =============================================================================

# Compare the probe, CAA, and direct optimisation vectors before steering
# Save plot-ready geometric results such as matched cosine, common-axis strength, etc.
python -u compare_steering_method_vectors.py \
  > out_compare_steering_method_vectors.txt 2>&1 

# Plot and summarise the pre-steering vector geometry
# This reads the comparison artifact above and creates the corresponding
# distributions, heatmaps, ordered checkpoint plots, and projection plots
python -u plot_steering_method_vectors.py --plotlist all \
  --output-dir exp34_steering_method_vector_plots \
  > out_plot_steering_method_vectors.txt 2>&1

# Evaluate all three vector types under the same steering setup 
CUDA_VISIBLE_DEVICES=0 python -u perform_steering.py \
  --suite all \
  --methods probe,caa,optimised \
  --save-validation-tokens \
  > out_perform_steering.txt 2>&1
status=$?
echo "" >> out_perform_steering.txt
echo "EXIT STATUS: $status" >> out_perform_steering.txt

# Plot and summarise the steering results
python -u plot_steering_results.py --plotlist all \
  --output-dir exp34_steering_results_plots \
  > out_plot_steering_results.txt 2>&1
status=$?
echo "" >> out_plot_steering_results.txt
echo "EXIT STATUS: $status" >> out_plot_steering_results.txt
## plot only commmand: python -u plot_steering_results.py --plotlist intervention_frequency_compact --output-dir exp34_steering_results_plots


# Focused sparse attention-head CAA ablation
CUDA_VISIBLE_DEVICES=0 python -u attention_head_ablations.py \
  > out_attention_head_ablations.txt 2>&1
status=$?
echo "" >> out_attention_head_ablations.txt
echo "EXIT STATUS: $status" >> out_attention_head_ablations.txt
## plot only commmand: python -u attention_head_ablations.py --plot-only > quick_out_attn_ablations_plot_only.txt 2>&1


# Extended comparison of attention-head steering methods
CUDA_VISIBLE_DEVICES=0 python -u compare_more_attention_head_steering.py \
  > out_compare_more_attention_head_steering.txt 2>&1
status=$?
echo "" >> out_compare_more_attention_head_steering.txt
echo "EXIT STATUS: $status" >> out_compare_more_attention_head_steering.txt
## plot only commmand: python -u compare_more_attention_head_steering.py --plot-only > quick_out_compare_more_plot_only.txt 2>&1


# [Optional] 
# Create tables: larger ones and then compact Overleaf versions
python -u make_steering_result_tables.py  --examples-per-effect 5 --output-dir exp34_steering_result_tables > out_make_steering_result_tables.txt 2>&1
python -u quick_latex_tables_and_samples.py > out_quick_latex_tables_and_samples.txt 2>&1