function [sinr] = sinr(gain, power_bs, num_user, num_bs)
    
    N0PSD = -174; % noise power spectrum density, dBm/Hz
	  var_awgn = 10.^((N0PSD-30)/10)*1*10^6; % awgn^2
    
    sinr = zeros(num_user, num_bs);
  for i = 1 : num_user
		for j = 1 : num_bs
			sinr(i, j) = power_bs(j) * gain(i, j) / (var_awgn.^2 + get_interf_numerical(power_bs, i, j, gain));
		end
	end

