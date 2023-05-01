    function lr = learning_rate_decay(alpha, epoch, dalpha, use_decay)
      % This function takes the initial learning rate, the current epoch,
      % the decay rate, and a boolean flag indicating whether to use the
      % decay or not as inputs, and returns the updated learning rate using
      % the following formula if use_decay is true:
      %
      %   lr = alpha / (1 + dalpha * epoch)
      %
      % If use_decay is false, the function returns the initial learning rate.

      if use_decay
          lr = alpha / (1 + dalpha * epoch);
      else
          lr = alpha;
      endif

    endfunction
