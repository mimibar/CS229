function [jVal,gradient]=costFunction(t)
    jVal = (t(1)-0.086898)^2 + (t(2)-0.092858)^2;
    gradient = zeros(2,1);
    gradient (1)=2*(t(1)-0.086898);
    gradient (2)=2*(t(2)-0.092858);