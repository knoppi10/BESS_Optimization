#!/bin/bash
# cleanup_project.sh
# Script to clean up the BESS_Optimization project for GitHub submission
# Run this script to remove old/redundant files

echo "🧹 Cleaning up BESS_Optimization project..."
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Files to delete - Old simulation scripts (replaced by Update.py)
OLD_SCRIPTS=(
    "bess_simulation.py"
    "bess_simulation_jan_mar_2019.py"
    "bess_simulation_quadratic.py"
    "bess_simulation_quadratic_sequential.py"
    "combine_charts.py"
    "analyze_costs.py"
    "analyze_slope.py"
    "gurobi_Test.py"
    "test_entsoe.py"
    "hurdle_rate_optimization.py"
)

# Old CSV files (replaced by _with_pt versions)
OLD_CSV=(
    "arbitrage_summary_jan_mar_2019.csv"
    "arbitrage_summary_multiyear.csv"
    "arbitrage_summary_quadratic.csv"
    "arbitrage_summary_quadratic_sequential.csv"
    "simulation_decisions_jan_mar_2019.csv"
    "simulation_decisions_multiyear.csv"
    "simulation_decisions_quadratic.csv"
    "simulation_decisions_quadratic_sequential.csv"
    "cost_analysis_detailed.csv"
    "cost_analysis_summary.csv"
    "parameter_study_impact_summary.csv"
    "parameter_study_summary.csv"
    "hurdle_rate_optimal_by_scenario.csv"
    "hurdle_rate_optimization_all_scenarios.csv"
)

# Old/redundant PNG files
OLD_PNG=(
    "combined_arbitrage_cycles.png"
    "combined_arbitrage_cycles_exact.png"
    "combined_arbitrage_cycles_large.png"
    "cost_breakdown.png"
    "plot_4_exogen_vs_endogen.png"
    "plot_spread_evolution.png"
    "hurdle_rate_optimization_all_scenarios.png"
    "plot_6_sensitivity_v2.png"
    "plot_7_sensitivity_matrix_v2.png"
    "plot_8_summary_table_v2.png"
)

# Other files to remove
OTHER_FILES=(
    ".DS_Store"
    "1-s2.0-S0140988325004359-main.pdf"
)

echo "📁 Removing old simulation scripts..."
for file in "${OLD_SCRIPTS[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Deleted: $file"
    fi
done

echo ""
echo "📊 Removing old CSV files..."
for file in "${OLD_CSV[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Deleted: $file"
    fi
done

echo ""
echo "🖼️  Removing old/redundant PNG files..."
for file in "${OLD_PNG[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Deleted: $file"
    fi
done

echo ""
echo "🗑️  Removing other unnecessary files..."
for file in "${OTHER_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Deleted: $file"
    fi
done

# Replace README with improved version
if [ -f "README_NEW.md" ]; then
    echo ""
    echo "📝 Updating README.md..."
    mv README_NEW.md README.md
    echo "  ✓ README.md updated"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining project structure:"
echo "----------------------------"
ls -la

echo ""
echo "📌 Next steps:"
echo "  1. Review the changes: git status"
echo "  2. Stage all changes: git add -A"
echo "  3. Commit: git commit -m 'Clean up project for submission'"
echo "  4. Push to GitHub: git push origin main"
