classdef pRF < handle
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
    % Population receptive field (pRF) mapping tool.
    %
    % prf = pRF(scan_params) creates an instance of the pRF class.
    % scan_params is a structure with 7 required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - n_samples : number of samples (volumes/TRs)
    %   - n_rows    : number of rows (in-plane resolution)
    %   - n_cols    : number of columns (in-plance resolution)
    %   - n_slices  : number of slices
    %   - w_stimulus: width of stimulus images in pixels
    %   - h_stimulus: height of stimulus images in pixels
    %
    % optional inputs are
    %   - hrf       : either a column vector containing a single hemodynamic response used for every voxel;
    %                 or a 4D matrix with a unique hemodynamic response per voxel.
    %                 By default the canonical two-gamma hemodynamic response function is generated internally 
    %                 based on the scan parameters.
    %
    % this class has the following functions
    %
    %   - hrf = pRF.get_hrf();
    %   - stimulus = pRF.get_stimulus();
    %   - tc = pRF.get_timecourses();
    %   - pRF.set_hrf(hrf);
    %   - pRF.set_stimulus(stimulus);
    %   - pRF.import_stimulus();
    %   - pRF.create_timecourses();
    %   - results = pRF.mapping(data);
    %
    % use help pRF.function to get more detailed help on any specific function (e.g. help pRF.mapping)
    %
    % typical workflow:
    % 1. prf = pRF(scan_params);
    % 2. prf.import_stimulus();
    % 3. prf.create_timecourses();
    % 4. results = prf.mapping(data);
    
    properties (Access = private)
        
        is
        
        % functions
        two_gamma               % two gamma hrf function
        gauss                   % isotropic Gaussian
        
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
        w_stimulus
        h_stimulus
        idx
        X
        Y
        Sigma
        hrf
        stimulus
        tc_fft
    end
    
    methods (Access = public)
        
        function self = pRF(scan_params,varargin)
            % constructor
            addpath(pwd)
            p = inputParser;
            addRequired(p,'scan_params',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(scan_params,varargin{:});
            
            self.is = 'pRF mapping tool';
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            self.gauss = @(mu_x,mu_y,sigma,X,Y) exp(-((X-mu_x).^2+...
                (Y-mu_y).^2)/(2*sigma^2));
            
            self.f_sampling = p.Results.scan_params.f_sampling;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = p.Results.scan_params.n_samples;
            self.n_rows = p.Results.scan_params.n_rows;
            self.n_cols = p.Results.scan_params.n_cols;
            self.n_slices = p.Results.scan_params.n_slices;
            self.n_total = self.n_rows*self.n_cols*self.n_slices;
            self.w_stimulus = p.Results.scan_params.w_stimulus;
            self.h_stimulus = p.Results.scan_params.h_stimulus;
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
            % returns the stimulus used by the class as a 3D matrix of dimensions width-by-height-by-time.
            stimulus = reshape(self.stimulus,self.h_stimulus,...
                self.w_stimulus,self.n_samples);
        end
        
        function tc = get_timecourses(self)
            % returns the timecourses predicted based on the effective stimulus and each combination of pRF model parameters as a time-by-combinations matrix.
            % Note that the predicted timecourses have not been convolved with a hemodynamic response.
            %
            tc = ifft(self.tc_fft);
        end
        
        function set_hrf(self,hrf)
            % replace the hemodynamic response with a new hrf column vector
            % or a 4-dimensional matrix with columns corresponding to time.
            % Dimensions 2, 3 and 4 need to match those of the BOLD data.
            if ndims(hrf)==4
                self.hrf = reshape(hrf,self.l_hrf,self.n_total);
            else
                self.hrf = hrf;
            end
        end
        
        function set_stimulus(self,stimulus)
            % provide a stimulus matrix.
            % This is useful if the .png files have already been imported and stored in matrix form.
            % Note that the provided stimulus matrix can be either 3D (width-by-height-by-time) or 2D (width*height-by-time).
            if ndims(stimulus==3)
                self.stimulus = reshape(stimulus,...
                    self.w_stimulus*self.h_stimulus,...
                    self.n_samples);
            else
                self.stimulus = stimulus;
            end
        end
        
        function import_stimulus(self)
            % imports the series of .png files constituting the stimulation protocol of the pRF experiment.
            % This series is stored internally in a matrix format (width-by-height-by-time).
            % The stimulus is required to generate timecourses.
            
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
        
        function create_timecourses(self,varargin)
            % creates predicted timecourses based on the effective stimulus and a range of isotropic receptive fields. 
            % Isotropic receptive fields are generated for a grid of location (x,y) and size parameters.
            %
            % optional inputs are
            %  - min_X      : lower bound of x location       (default = -10.0)
            %  - max_X      : upper bound of x location       (default =  10.0)
            %  - number_X   : steps from lower to upper bound (default =  30.0)
            %  - min_Y      : lower bound of y location       (default = -10.0)
            %  - max_Y      : upper bound of y location       (default =  10.0)
            %  - number_Y   : steps from lower to upper bound (default =  30.0)
            %  - min_size   : lower bound of RF size          (default =   0.2)
            %  - max_size   : upper bound of RF size          (default =   7.0)
            %  - number_size: steps from lower to upper bound (default =  10.0)
            
            text = 'creating timecourses...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            p = inputParser;
            addOptional(p,'number_X',30);
            addOptional(p,'min_X',-10);
            addOptional(p,'max_X',10);
            addOptional(p,'number_Y',30);
            addOptional(p,'min_Y',-10);
            addOptional(p,'max_Y',10);
            addOptional(p,'number_size',10);
            addOptional(p,'min_size',0.2);
            addOptional(p,'max_size',7);
            p.parse(varargin{:});
            
            n_x = p.Results.number_X;
            min_x = p.Results.min_X;
            max_x = p.Results.max_X;
            n_y = p.Results.number_Y;
            min_y = p.Results.min_Y;
            max_y = p.Results.max_Y;
            n_size = p.Results.number_size;
            min_size = p.Results.min_size;
            max_size = p.Results.max_size;
            self.n_points = n_x*n_y*n_size;
            
            X_ = ones(self.h_stimulus,1) * linspace(min_x,...
                max_x,self.w_stimulus);
            Y_ = -linspace(min_y,max_y,...
                self.h_stimulus)' * ones(1,self.w_stimulus);
            
            X_ = reshape(X_,self.w_stimulus*self.h_stimulus,1);
            Y_ = reshape(Y_,self.w_stimulus*self.h_stimulus,1);
            
            i = (0:self.n_points-1)';
            self.idx = [floor(i/(n_y*n_size))+1,...
                mod(floor(i/(n_size)),n_y)+1,...
                mod(i,n_size)+1];
            self.X = linspace(min_x,max_x,n_x);
            self.Y = linspace(min_y,max_y,n_y);
            self.Sigma = linspace(min_size,max_size,n_size);
            
            W = single(zeros(self.n_points,...
                self.w_stimulus*self.h_stimulus));
            for j=1:self.n_points
                x = self.X(self.idx(j,1));
                y = self.Y(self.idx(j,2));
                sigma = self.Sigma(self.idx(j,3));
                W(j,:) = self.gauss(x,y,sigma,X_,Y_)';
                waitbar(j/self.n_points*.9,wb);
            end
            
            tc = W * self.stimulus;
            waitbar(1,wb);
            self.tc_fft = fft(tc');
            close(wb)
            fprintf('done\n');
        end
        
        function results = mapping(self,data,varargin)
            % identifies the best fitting timecourse for each voxel and
            % returns a structure with the following fields
            %  - R
            %  - X
            %  - Y
            %  - Sigma
            %  - Eccentricity
            %  - Polar_Angle
            %
            % each field is 3-dimensional corresponding to the volumetric dimensions of the data.
            %
            % required inputs are
            %  - data     : a 4-dimensional matrix of empirically observed
            %                BOLD timecourses. Columns correspond to time
            %                (volumes/TRs).
            % optional inputs are
            %  - threshold: minimum voxel intensity (default = 100.0)
            
            text = 'mapping population receptive fields...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            p = inputParser;
            addRequired(p,'data',@isnumeric);
            addOptional(p,'threshold',100);
            p.parse(data,varargin{:});
            
            data = p.Results.data;
            threshold = p.Results.threshold;
            
            [~,numX,numY,numZ] = size(data);
            numXYZ = numX * numY *numZ;
            data = reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,numXYZ);
            mean_signal = mean(data);
            data = zscore(data);
            mag_d = sqrt(sum(data.^2));
            
            results.R = zeros(numXYZ,1);
            results.X = NaN(numXYZ,1);
            results.Y = NaN(numXYZ,1);
            results.Sigma = NaN(numXYZ,1);
            
            if size(self.hrf,2)==1
                hrf_fft = fft(repmat([self.hrf;...
                    zeros(self.n_samples-self.l_hrf,1)],...
                    [1,self.n_points]));
                tc = zscore(ifft(self.tc_fft.*hrf_fft))';
                mag_tc = sqrt(sum(tc.^2,2));
                for v=1:numXYZ
                    if mean_signal(v)>threshold
                        CS = (tc*data(:,v))./...
                            (mag_tc*mag_d(v));
                        id = isinf(CS) | isnan(CS);
                        CS(id) = 0;
                        [results.R(v),j] = max(CS);
                        results.X(v) = self.X(self.idx(j,1));
                        results.Y(v) = self.Y(self.idx(j,2));
                        results.Sigma(v) = self.Sigma(self.idx(j,3));
                    end
                    waitbar(v/numXYZ,wb)
                end
            else
                hrf_fft_all = fft([self.hrf;...
                            zeros(self.n_samples-self.l_hrf,numXYZ)]);
                for v=1:numXYZ
                    if mean_signal(v)>threshold
                        tic
                        hrf_fft = repmat(hrf_fft_all(:,v),...
                            [1,self.n_points]);
                        tc = zscore(ifft(self.tc_fft.*hrf_fft))';
                        mag_tc = sqrt(sum(tc.^2,2));
                        
                        CS = (tc*data(:,v))./...
                            (mag_tc*mag_d(v));
                        id = isinf(CS) | isnan(CS);
                        CS(id) = 0;
                        [results.R(v),j] = max(CS);
                        results.X(v) = self.X(self.idx(j,1));
                        results.Y(v) = self.Y(self.idx(j,2));
                        results.Sigma(v) = self.Sigma(self.idx(j,3));
                        toc
                    end
                    waitbar(v/numXYZ,wb)
                end
            end
            results.R = reshape(results.R,numX,numY,numZ);
            results.X = reshape(results.X,numX,numY,numZ);
            results.Y = reshape(results.Y,numX,numY,numZ);
            results.Sigma = reshape(results.Sigma,numX,numY,numZ);
            results.Eccentricity = abs(results.X+results.Y*1i);
            results.Polar_Angle = angle(results.X+results.Y*1i);
            close(wb)
            fprintf('done\n');
        end
    end
end