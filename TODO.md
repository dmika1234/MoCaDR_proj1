# Two-dimensional minimization: r, col_weight
## SVD1
- r: 5:20 i potem co 5 do 50
- weights: od 0.2 do 0.6
## SVD2
- r: 
- weights: od 0.2 do 0.6

## NMF
- r: 5, 10, 15, 15:55
- weights: od 0.2 do 0.6

# Useful code
geom_hline(yintercept=best_rmse, linetype='dashed', color='black') + \
    geom_vline(xintercept=best_r, linetype='dashed', color='black') + \
    geom_label(aes(x=float('inf'), y=float('inf')), label=f'RMSE={str(np.round(best_rmse, 5))} for r={str(best_r)}', va="top", ha="right") + \