## prob 67

$$ \begin{aligned}
\min\quad & C\_out + x10\\
\text{Subject to} \quad & -υ\_cumulative\_in + υ\_cumulative\_out - υ = 0.0\\
 & -0.10999731826274792 S\_out - 6.72435124018722 I\_out - 32.99999980160262 C\_out - 3.935143418913463e-8 υ\_cumulative\_out + x10 \geq -0.01864746859508025\\
 & -0.7096312399694777 S\_out - 17.76530422467338 I\_out - 32.999999799355486 C\_out - 0.02143535417699752 υ\_cumulative\_out + x10 \geq -0.3413010474330538\\
 & -0.7289886791921243 S\_out - 17.72580788644358 I\_out - 32.99999979937917 C\_out - 0.03595943069622046 υ\_cumulative\_out + x10 \geq -0.48980422122209927\\
 & -0.7461025533675305 S\_out - 17.672921817589014 I\_out - 32.999999799396285 C\_out - 0.27449518884532154 υ\_cumulative\_out + x10 \geq -2.8778825939738586\\
 & -0.7482002032679584 S\_out - 17.684432736473276 I\_out - 32.99999979939292 C\_out - 0.2700972338207066 υ\_cumulative\_out + x10 \geq -2.834345218229531\\
 & υ\_cumulative\_out \leq 10.0\\
 & S\_in = 0.17724517293061848\\
 & I\_in = 0.007283760735628131\\
 & C\_in = 0.8127548269262713\\
 & υ\_cumulative\_in = 10.000000089036952\\
 & S\_out \geq 0.0\\
 & I\_out \geq 0.0\\
 & C\_out \geq 0.0\\
 & υ\_cumulative\_out \geq 0.0\\
 & υ \geq 0.0\\
 & x10 \geq 0.0\\
 & υ \leq 0.5\\
 & (S_out - (S_in - (1.0 - exp(-((1.0 - υ)) * 0.5 * I_in * 1.0)) * S_in)) - 0.0 = 0\\
 & (I_out - ((I_in + (1.0 - exp(-((1.0 - υ)) * 0.5 * I_in * 1.0)) * S_in) - (1.0 - exp(-0.25 * 1.0)) * I_in)) - 0.0 = 0\\
 & (C_out - (C_in + (1.0 - exp(-((1.0 - υ)) * 0.5 * I_in * 1.0)) * S_in)) - 0.0 = 0\\
\end{aligned} $$

## prob 22

$$ \begin{aligned}
\min\quad & C\_out + x10\\
\text{Subject to} \quad & -υ\_cumulative\_in + υ\_cumulative\_out - υ = 0.0\\
 & υ\_cumulative\_in \leq 10.0\\
 & S\_in = 0.9272614125085762\\
 & I\_in = 0.014811515433252053\\
 & C\_in = 0.06273858749142384\\
 & υ\_cumulative\_in = 10.499988104056724\\
 & S\_out \geq 0.0\\
 & I\_out \geq 0.0\\
 & C\_out \geq 0.0\\
 & υ\_cumulative\_out \geq 0.0\\
 & υ \geq 0.0\\
 & x10 \geq 0.0\\
 & υ \leq 0.5\\
 & (S_out - (S_in - (1.0 - exp(-((1.0 - υ)) * 0.5 * I_in * 1.0)) * S_in)) - 0.0 = 0\\
 & (I_out - ((I_in + (1.0 - exp(-((1.0 - υ)) * 0.5 * I_in * 1.0)) * S_in) - (1.0 - exp(-0.25 * 1.0)) * I_in)) - 0.0 = 0\\
 & (C_out - (C_in + (1.0 - exp(-((1.0 - υ)) * 0.5 * I_in * 1.0)) * S_in)) - 0.0 = 0\\
\end{aligned} $$