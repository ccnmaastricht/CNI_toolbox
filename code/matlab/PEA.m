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
    % phase-encoding analysis tool.
    %
    % pea = PEA(parameters) creates an instance of the PEA class.
    %
    % parameters is a structure with 7 required fields
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
    % 1. pea = PEA(parameters);
    % 2. pea.set_delay(delay);
    % 3. pea.set_direction(direction);
    % 4. results = pea.fitting(data);
    
    
    properties (Access = private)
        
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
        
        function self = PEA(parameters)
            % constructor
            
            self.f_sampling = parameters.f_sampling;
            self.f_stim = parameters.f_stim;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = parameters.n_samples;
            self.n_rows = parameters.n_rows;
            self.n_cols = parameters.n_cols;
            self.n_slices = parameters.n_slices;
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
        
        function results = fitting(self,data,varargin)
            % identifies phase and ampltitude at stimulation frequency for
            % each voxel and returns a structure with the following fields
            %  - phase
            %  - ampltitude
            %  - f_statistic
            %  - p_value
            %
            % the dimension of each field corresponds to the dimensions of
            % the data.
            %
            % required inputs are
            %  - data  : a tensor of empirically observed BOLD timecourses
            %            whose rows correspond to time (volumes).
            %
            % optional inputs are
            %  - threshold: minimum voxel intensity (default = 100.0)
            %  - mask     : binary mask for selecting voxels
            
            progress('performing phase encoding analysis')
            
            p = inputParser;
            addRequired(p,'data',@isnumeric);
            addOptional(p,'threshold',100);
            addOptional(p,'mask',[]);
            p.parse(data,varargin{:});
            
            data = single(p.Results.data);
            threshold = p.Results.threshold;
            mask = p.Results.mask;
            
            F = exp(self.direction*2i*pi*self.f_stim*(self.t-self.delay));
            X = zscore([real(F)',imag(F)']);
            
            data = reshape(single(data(1:self.n_samples,:,:,:)),...
                self.n_samples,self.n_total);
            
            mean_signal = mean(data);
            sdev_signal = std(data);
            
            if isempty(mask)
                mask = mean_signal>=threshold;
            end
            mask = logical(mask(:));
            mask = mask & (sdev_signal(:) > 0);
            voxel_index = find(mask);
            n_voxels = numel(voxel_index);
            
            data = zscore(data(:, mask));
            
            beta = (X' * X) \ X' * data;
            
            Y_ = X * beta;
            residuals = data - Y_;
            
            results.phase = zeros(self.n_total,1);
            results.ampltitude = zeros(self.n_total,1);
            results.F_stat = zeros(self.n_total,1);
            results.p_value = ones(self.n_total,1);
            
            df1 = 1;
            df2 = self.n_samples-2;
            for m=1:n_voxels
                v = voxel_index(m);
                
                % estimate and correct for autocorrelation
                T = [[0;residuals(1:end-1,m)],[zeros(2,1); residuals(1:end-2,m)]];
                W = [1; -((T'* T) \ T' * residuals(:,m))];
                Xc(:,1) = [X(:,1), [0;X(1:end-1,1)], [zeros(2,1);X(1:end-2,1)]] * W;
                Xc(:,2) = [X(:,2), [0;X(1:end-1,2)], [zeros(2,1);X(1:end-2,2)]] * W;
                Dc = [data(:,m), [0;data(1:end-1,m)], [zeros(2,1);data(1:end-2,m)]] * W;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                b = (Xc' * Xc) \ Xc' * Dc;
                y = Xc * b;
                mu = mean(Dc);
                MSM = (y-mu)'*(y-mu)/df1;
                MSE = (y-Dc)'*(y-Dc)/df2;
                
                results.phase(v) = angle(b(1)+b(2)*1i);
                results.ampltitude(v) = abs(b(1)+b(2)*1i);
                results.F_stat(v) = MSM/MSE;
                results.p_value(v) = max(1-fcdf(MSM/MSE,df1,df2),1e-20);
                
                progress(m / n_voxels * 20)
            end
            
            results.phase = reshape(results.phase,...
                self.n_rows,self.n_cols,self.n_slices);
            results.ampltitude = reshape(results.ampltitude,...
                self.n_rows,self.n_cols,self.n_slices);
            results.F_stat = reshape(results.F_stat,...
                self.n_rows,self.n_cols,self.n_slices);
            results.p_value = reshape(results.p_value,...
                self.n_rows,self.n_cols,self.n_slices);
            
        end
        
    end
    
end