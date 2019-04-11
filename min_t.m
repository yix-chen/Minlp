function [c,ceq] = min_t(x)
	% sca
    load('const.mat')
    load('alphaandbeta.mat','alpha','beta')

    n_d = length(x_d);
	throughput = zeros(num_user, 1);
	gma = zeros(num_user, num_bs);

	for i = 1 : num_user
		for j = 1 : num_bs
            interf = get_interf(x, x_d, i, j, gain);
            gma(i, j) = exp(x(num_user * num_bs + j - n_d)) * gain(i, j) / (var_awgn.^2 + interf);
            gma(i, j) = log2(gma(i, j));
		end
	end

	for i = 1 : num_user
		for j = 1 : num_bs
			idx = (i - 1) * num_bs + j;
			if idx <= n_d
				throughput(i) = throughput(i) + x_d(idx) * bandw * (alpha(i, j) * gma(i, j) + beta(i,j));
			else
				throughput(i) = throughput(i) + x(idx - n_d) * bandw * (alpha(i, j) * gma(i, j) + beta(i,j));
			end
		end
	end

	mint = ones(num_user, 1);
	mint = mint .* x(end);
	c = mint - throughput;
	ceq = [];