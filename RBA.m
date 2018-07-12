classdef RBA < handle
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
    % rba = RBA(params) creates an instance of the RBA class.
    % params is a structure with 7 required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - n_samples : number of samples (volumes)
    %   - n_rows    : number of rows (in-plane resolution)
    %   - n_cols    : number of columns (in-plance resolution)
    %   - n_slices  : number of slices
    %
    % this class has the following function
    %
    %   - delay = RBA.get_delay();
    %   - direction = RBA.get_direction(direction);
    %   - RBA.set_delay(delay);
    %   - RBA.set_direction(direction);
    %   - results = RBA.fitting(data);
    %
    % use help RBA.function to get more detailed help on any specific
    % function (e.g. help RBA.fitting)
    %
    % typical workflow:
    % 1. rba = RBA(params);
    % 2. rba.set_delay(delay);
    % 3. rba.set_direction(direction);
    % 4. results = rba.perform_ridge(data,lambda);
    
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
        
        function self = RBA(params,varargin)
            % constructor
            addpath(pwd)
            p = inputParser;
            addRequired(p,'params',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(params,varargin{:});
            
            self.is = 'RBA tool';
            
            
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
        
        function stimulus = get_stimulus(self)
            % returns the stimulus used by the class as a 3D matrix of dimensions height-by-width-by-time.
            stimulus = reshape(self.stimulus,self.h_stimulus,...
                self.w_stimulus,self.n_samples);
        end
        
        
        function design = get_design(self)
            % returns the design matrix.
            design = self.X;
        end
        
        function filter = get_pyramid(self)
            % not ready
           filter = self.filter; 
        end
        
        function set_hrf(self,hrf)
            % replace the hemodynamic response with a new hrf column vector.
            self.l_hrf = size(hrf,1);
            self.hrf = hrf;
            
        end
        
        function set_stimulus(self,stimulus)
            % provide a stimulus matrix.
            % This is useful if the .png files have already been imported and stored in matrix form.
            % Note that the provided stimulus matrix can be either 3D (height-by-width-by-time) or 2D (height*width-by-time).
            if ndims(stimulus==3)
                self.stimulus = reshape(stimulus,...
                    self.w_stimulus*self.h_stimulus,...
                    self.n_samples);
            else
                self.stimulus = stimulus;
            end
        end
        
        function set_design(self,X)
            % provide a t-by-p design matrix to the class with t timepoints
            % and p predictors.
            self.X = X;
        end
        
        function import_stimulus(self)
            % imports the series of .png files constituting the stimulation protocol of the pRF experiment.
            % This series is stored internally in a matrix format (height-by-width-by-time).
            % The stimulus is required to generate the design matrix.
            
            [~,path] = uigetfile('*.png',...
                'Please select the first .png file');
            files = dir([path,'*.png']);
            text = 'loading stimulus...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            im = imread([path,files(1).name]);
            self.stimulus = zeros(self.h_stimulus,self.w_stimulus,...
                self.n_samples);
            self.stimulus(:,:,1) = im(:,:,1);
            l = regexp(files(1).name,'\d')-1;
            prefix = files(1).name(1:l);
            name = {files.name};
            str  = sprintf('%s#', name{:});
            num  = sscanf(str, [prefix,'%d.png#']);
            [~, index] = sort(num);
            
            for t=2:self.n_samples
                im = imread([path,files(index(t)).name]);
                self.stimulus(:,:,t) = im(:,:,1);
                waitbar(t/self.n_samples,wb)
            end
            mn = min(self.stimulus(:));
            range = max(self.stimulus(:))-mn;
            
            self.stimulus = (reshape(self.stimulus,...
                self.w_stimulus*self.h_stimulus,...
                self.n_samples)-mn)/range;
            close(wb)
            fprintf('done\n');
        end
        
        function create_pyramid(self)
            % not ready
            sf = 2.^(1:5);
            a = linsapce(0,7/8*pi,8);
            
        end
        
        function create_design(self)
            % create a t-by-p design matrix to the class with t timepoints
            % and p predictors based on filter responses to image stimuli
        end
        
        
        
        function results = perform_ridge(self,data,lambda)
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
            x = zscore(ifft(X_fft.*hrf_fft));
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