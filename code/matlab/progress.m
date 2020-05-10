function progress(c)

if isnumeric(c)
    i = round(c);
    fprintf(1,'\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b[%s%s] %3d%%\n',...
        repmat('=',1,i),repmat(' ',1,20-i), uint8(i/20*100))
    
else
    fprintf(1,'%s\n%s',c, repmat(' ', 28,1))
end

