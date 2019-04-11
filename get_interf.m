%% get_interf: function description
function [interf] = get_interf(x, x_d, i, j, gain)
	[num_user, num_bs] = size(gain);

	n_d = length(x_d);
	interf = 0;
	for k = 1 : num_bs
		interf = interf + exp(x(num_user * num_bs + k - n_d)) * gain(i, k);
	end

	interf = interf - exp(x(num_user * num_bs + j - n_d)) * gain(i, j);
	

