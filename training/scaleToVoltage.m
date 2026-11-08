function Y = scaleToVoltage(X)
%#codegen
% tanh output (-1 to 1) to motor voltage (-6 to 6 V)
Y = 6 .* X;
end
