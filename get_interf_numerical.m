%% get_interf: function description
function [interf] = get_interf_numerical(p, i, j, gain)
	[num_user, num_bs] = size(gain);

	interf = 0;
	for k = 1 : num_bs
		interf = interf + p(j) * gain(i, k);
	end

	interf = interf - p(j) * gain(i, j);
	

