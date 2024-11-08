function is_match = verify_match(n_in, n_out)

alpha = 8;
beta = 0.3;
is_match = ((1-beta)*n_in - beta*n_out - alpha > 0);

end

