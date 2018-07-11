classdef MF < handle
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
    % Model free mapping tool.
    %
    % mf = MF(params) creates an instance of the MF class.
    % params is a structure with 7 required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - n_samples : number of samples (volumes)
    %   - n_rows    : number of rows (in-plane resolution)
    %   - n_cols    : number of columns (in-plance resolution)
    %   - n_slices  : number of slices
    %
    % this class has the following function
    %
    %   - delay = MF.get_delay();
    %   - direction = MF.get_direction(direction);
    %   - MF.set_delay(delay);
    %   - MF.set_direction(direction);
    %   - results = MF.fitting(data);
    %
    % use help MF.function to get more detailed help on any specific
    % function (e.g. help MF.fitting)
    %
    % typical workflow:
    % 1. mf = MF(params);
    % 2. mf.set_delay(delay);
    % 3. mf.set_direction(direction);
    % 4. results = mf.ridge(data,lambda);
    
    
    properties (Access = private)
        
        is
        % functions
        two_gamma               % two gamma hrf function
        
        % parameters
        f_sampling
        p_sampling
        n_samples
        n_points
        n_rows
        n_cols
        n_slices
        n_total
        l_hrf
        hrf
        X
        
    end
    
    methods (Access = public)
        
        function self = MF(params,varargin)
            % constructor
            addpath(pwd)
            p = inputParser;
            addRequired(p,'params',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(params,varargin{:});
            
            self.is = 'MF tool';
            
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            
            self.f_sampling = params.f_sampling;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = params.n_samples;
            self.n_rows = params.n_rows;
            self.n_cols = params.n_cols;
            self.n_slices = params.n_slices;
            self.n_total = self.n_rows*self.n_cols*self.n_slices;
            self.X = [];
            
            if ~isempty(p.Results.hrf)
                self.l_hrf = size(p.Results.hrf,1);
                if ndims(p.Results.hrf)==4
                    self.hrf = reshape(p.Results.hrf,...
                        self.l_hrf,self.n_total);
                else
                    self.hrf = p.Results.hrf;
                end
                
            else
                self.hrf = self.two_gamma(0:self.p_sampling:34)';
                self.l_hrf = numel(self.hrf);
            end
            
        end
        
        
        function hrf = get_hrf(self)
            % returns the hemodynamic response used by the class.
            % If a single hrf is used for every voxel, this function returns a column vector.
            % If a unique hrf is used for each voxel, this function returns a 4-dimensional matrix with columns corresponding to time.
            if size(self.hrf,2)>1
                hrf = reshape(self.hrf,self.l_hrf,...
                    self.n_rows,self.n_cols,self.n_slices);
            else
                hrf = self.hrf;
            end
        end
        
        function design = get_design(self)
            % returns the design matrix.
            design = self.X;
        end
        
        function set_hrf(self,hrf)
            % replace the hemodynamic response with a new hrf column vector.
            self.l_hrf = size(hrf,1);
            self.hrf = hrf;
            
        end
        
        function set_design(self,X)
            % provide a t-by-p design matrix to the class with t timepoints
            % and p predictors.
            self.X = X;
        end
        
        function results = ridge(self,data,lambda)
            % performs ridge regression and returns a structure with the following fields
            %  - Beta
            %  - RSS
            %  - F_statistic
            %  - P_value
            %
            % the dimension of each field corresponds to the dimension of the data.
            %
            % required inputs are
            %  - data     : a 4-dimensional matrix of empirically observed BOLD timecourses.
            %               Columns correspond to time (volumes).
            %  - lambda   : shrinkage parameter
            
            text = 'performing ridge regression...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            p = size(self.X,2);
            X_fft = fft(self.X);
            hrf_fft = fft(repmat([self.hrf;...
                    zeros(self.n_samples-self.l_hrf,1)],...
                    [1,p]));
            x = zscore(X_fft.*hrf_fft);
            [U,D,V] = svd(x,'econ');
            XX = V * inv(D^2 + lambda * eye(self.n_samples)) * D * U';
            
            data = zscore(reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total));
            
            results.Beta = single(zeros(p,self.n_total));
            results.RSS = zeros(self.n_total,1);
            results.F_stat = zeros(self.n_total,1);
            results.P_value = zeros(self.n_total,1);
            
            df1 = 2;
            df2 = self.n_samples-1;
            for v=1:self.n_total
                b = XX * data(:,v);
                y = x * b;
                y_ = mean(y);
                MSM = (y-y_)'*(y-y_)/df1;
                MSE = (y-data(:,v))'*(y-data(:,v))/df2;
                
                results.Beta(:,v) = b;
                results.RSS(v) = (y-data(:,v))'*(y-data(:,v));
                results.F_stat(v) = MSM/MSE;
                results.P_value(v) = 1-fcdf(MSM/MSE,df1,df2);
                waitbar(v/self.n_total,wb)
            end
            
            results.Beta = reshape(results.Beta,p,...
                self.n_rows,self.n_cols,self.n_slices);
            results.RSS = reshape(results.RSS,...
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