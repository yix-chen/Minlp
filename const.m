%% const: function description
function [outputs] = const(num_user, num_bs, gain, power_bs, bandw, x_d)

	N0PSD = -174; % noise power spectrum density, dBm/Hz
	var_awgn = 10.^((N0PSD-30)/10)*1*10^6; % awgn^2
	save const.mat

	outputs = var_awgn;