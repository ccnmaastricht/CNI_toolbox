classdef PEA < handle
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%                               LICENSE                             %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Copyright 2018 Mario Senden
    %
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU Lesser General Public License as published
    % by the Free Software Foundation, either version 3 of the License, or
    % (at your option) any later version.
    %
    % This program is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    % GNU Lesser General Public License for more details.
    %
    % You should have received a copy of the GNU Lesser General Public License
    % along with this program.  If not, see <http://www.gnu.org/licenses/>.
    %
    %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%                             DESCRIPTION                           %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Phase-encoding analysis tool.
    %
    % pea = PEA(params) creates an instance of the PEA class.
    % params is a structure with 7 required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - f_stim    : stimulation frequency
    %   - n_samples : number of samples (volumes)
    %   - n_rows    : number of rows (in-plane resolution)
    %   - n_cols    : number of columns (in-plance resolution)
    %   - n_slices  : number of slices
    %
    % this class has the following function
    %
    %   - delay = PEA.get_delay();
    %   - direction = PEA.get_direction();
    %   - PEA.set_delay(delay);
    %   - PEA.set_direction(direction);
    %   - results = PEA.fitting(data);
    %
    % use help PEA.function to get more detailed help on any specific
    % function (e.g. help PEA.fitting)
    %
    % typical workflow:
    % 1. pea = PEA(params);
    % 2. pea.set_delay(delay);
    % 3. pea.set_direction(direction);
    % 4. results = pea.fitting(data);
    
    
    properties (Access = private)
        
        is
        
        % parameters
        f_sampling
        f_stim
        p_sampling
        n_samples
        n_points
        n_rows
        n_cols
        n_slices
        n_total
        delay
        direction
        t
        
    end
    
    methods (Access = public)
        
        function self = PEA(params)
            % constructor
            self.is = 'PEA tool';
            
            self.f_sampling = params.f_sampling;
            self.f_stim = params.f_stim;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = params.n_samples;
            self.n_rows = params.n_rows;
            self.n_cols = params.n_cols;
            self.n_slices = params.n_slices;
            self.n_total = self.n_rows*self.n_cols*self.n_slices;
            self.t = (0:self.p_sampling:self.p_sampling*self.n_samples-1);
            self.delay = 0;
            self.direction = 1;
        end
        
        function delay = get_delay(self)
            % returns the delay used by the class.
            delay = self.delay;
        end
        
        function direction = get_direction(self)
            % returns the direction used by the class.
            if isnumeric(self.direction)
                if self.direction<0
                    direction = 'cw';
                else
                    direction = 'ccw';
                end
            else
                direction = self.direction;
            end
        end
        
        function set_delay(self,delay)
            % provide a delay for the class.
            self.delay = delay;
        end
        
        function set_direction(self,direction)
            % provide a direction of motion for the class.
            % This can be either numeric (-1,1) or in form of a string ('cw','ccw')
            % for clockwise and counterclockwise rotation, respectively.
            % Contracting rings are also considered to move clockwise (-1)
            % while expanding rings are considered to move counterclockwise (1).
            if isnumeric(direction)
                self.direction = direction;
            else
                if strcmp(direction,'cw')
                    self.direction = -1;
                else
                    self.direction = 1;
                end
                
            end
        end
        
        function results = fitting(self,data)
            % identifies phase and amplitude at stimulation frequency for
            % each voxel and returns a structure with the following fields
            %  - Phase
            %  - Amplitude
            %  - F_statistic
            %  - P_value
            %
            % the dimension of each field corresponds to the dimensions of
            % the data.
            %
            % required inputs are
            %  - data  : a matrix of empirically observed BOLD timecourses
            %            whose columns correspond to time (volumes).
            
            text = 'performing phase encoding analysis...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            F = exp(self.direction*2i*pi*self.f_stim*(self.t-self.delay));
            X = zscore([real(F)',imag(F)']);
            XX = (X'*X)\X';
            data = zscore(reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total));
            std_signal = std(data);
            results.Phase = zeros(self.n_total,1);
            results.Amplitude = zeros(self.n_total,1);
            results.F_stat = zeros(self.n_total,1);
            results.P_value = ones(self.n_total,1);
            
            df1 = 2;
            df2 = self.n_samples-1;
            for v=1:self.n_total
                if std_signal>0
                    b = XX * data(:,v);
                    y = X*b;
                    
                    % estimate and correct for autocorrelation
                    r = y-data(:,v);
                    T = [[0;r(1:end-1)],[zeros(2,1);r(1:end-2)]];
                    W = [1; -((T'* T) \ T' * r)];
                    Xc(:,1) = [X(:,1),[0;X(1:end-1,1)],[zeros(2,1);X(1:end-2,1)]] * W;
                    Xc(:,2) = [X(:,2),[0;X(1:end-1,2)],[zeros(2,1);X(1:end-2,2)]] * W;
                    Dc = [data(:,v),[0;data(1:end-1,v)],[zeros(2,1);data(1:end-2,v)]] * W;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    b = (Xc' * Xc) \ Xc' * Dc;
                    y = Xc * b;
                    y_ = mean(y);
                    MSM = (y-y_)'*(y-y_)/df1;
                    MSE = (y-data(:,v))'*(y-Dc)/df2;
                    
                    results.Phase(v) = angle(b(1)+b(2)*1i);
                    results.Amplitude(v) = abs(b(1)+b(2)*1i);
                    results.F_stat(v) = MSM/MSE;
                    results.P_value(v) = max(1-fcdf(MSM/MSE,df1,df2),1e-20);
                end
                waitbar(v/self.n_total,wb)
            end
            
            results.Phase = reshape(results.Phase,...
                self.n_rows,self.n_cols,self.n_slices);
            results.Amplitude = reshape(results.Amplitude,...
                self.n_rows,self.n_cols,self.n_slices);
            results.F_stat = reshape(results.F_stat,...
                self.n_rows,self.n_cols,self.n_slices);
            results.P_value = reshape(results.P_value,...
                self.n_rows,self.n_cols,self.n_slices);
            
            close(wb)
            fprintf('done\n');
        end
        
    end
    
end