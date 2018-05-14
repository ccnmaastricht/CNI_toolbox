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
    % scan_params is a structure with three required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - n_samples : number of samples (volumes)
    %   - r_image   : x/y resolution of (square) stimulus images. This
    %                 corresponds to width & height of stimulus images in
    %                 pixels.
    %
    % optional inputs are
    %   - hrf       : a column vector containing a hemodynamic response.
    %                 This response is then used for every voxels.        
    %                 Alternatively, a time-by-voxels matrix with a unique 
    %                 hemodynamic response per voxel can be provided.
    %                 By default a two-gamma hemodynamic response function
    %                 is generated internally given the scan parameters.
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
    % use help pRF.function to get more detailed help on any specific
    % function (e.g. help pRF.mapping)
    %
    % typical workflow:
    % 1. prf = pRF(scan_params);
    % 2. prf.import_stimulus();
    % 3. prf.create_timecourses();
    % 4. results = prf.mapping(data);
    
    properties (Access = private)
        % functions
        two_gamma               % two gamma hrf function
        gauss                   % isotropic Gaussian
        
        % parameters
        f_sampling
        p_sampling
        n_samples
        n_points
        l_hrf
        r_image
        idx
        XY
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
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            self.gauss = @(mu_x,mu_y,sigma,X,Y) exp(-((X-mu_x).^2+...
                (Y-mu_y).^2)/(2*sigma^2));
            
            self.f_sampling = p.Results.scan_params.f_sampling;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = p.Results.scan_params.n_samples;
            self.r_image = p.Results.scan_params.r_image;
            if ~isempty(p.Results.hrf)
                self.hrf = p.Results.hrf;
                self.l_hrf = size(self.hrf,1);
            else
                self.hrf = self.two_gamma(0:self.p_sampling:34)';
                self.l_hrf = numel(self.hrf);
            end
            
        end
        
        function hrf = get_hrf(self)
            % returns the hemodynamic response used by the class. 
            % If a single hrf is used for every voxel, this function
            % returns a column vector.
            % If a unique hrf is used for each voxel, this function returns
            % a time-by-voxels matrix.
            hrf = self.hrf;
        end
        
        function stimulus = get_stimulus(self)
            % returns the stimulus used by the class as a 3D matrix of
            % dimensions width-by-height-by-time.
           stimulus = reshape(self.stimulus,self.r_image,...
               self.r_image,self.n_samples); 
        end
        
        function tc = get_timecourses(self)
            % returns the timecourses predicted based on the effective
            % stimulus and each combination of pRF model parameters as a
            % time-by-combinations matrix. Note that the predicted
            % timecourses have not been convolved with a hemodynamic
            % response.
            % 
           tc = ifft(self.tc_fft); 
        end
        
        function set_hrf(self,hrf)
            % replace the hemodynamic resonse with a new hrf column vector
            % or time-by-voxels hrf matrix.
            self.hrf = hrf;
        end
        
        function set_stimulus(self,stimulus)
            % provide a stimulus matrix. 
            % This is useful if the .png files have already been imported 
            % and stored in matrix form. 
            % Note that the provided stimulus matrix can be either 3D 
            % (width-by-height-by-time) or 2D (width*height-by-time).
            if ndims(stimulus==3)
            self.stimulus = reshape(stimulus,self.r_image^2,...
                self.n_samples);
            else
                self.stimulus = stimulus;
            end
        end

        function import_stimulus(self)
            % imports the series of .png files constituting the stimulation
            % protocol of the pRF experiment and stores them internally in
            % a matrix format (width-by-height-by-time). 
            % The stimulus is required to generate timecourses.
        
            [~,path] = uigetfile('*.png',...
                'Please select the first .png file');
            files = dir([path,'*.png']);
            
            wb = waitbar(0,'loading stimulus...',...
                'Name','pRF mapping tool');
            
            im = imread([path,files(1).name]);
            self.stimulus = false(self.r_image,self.r_image,...
                self.n_samples);
            self.stimulus(:,:,1) = logical(im(:,:,1));
            l = regexp(files(1).name,'\d')-1;
            prefix = files(1).name(1:l);
            name = {files.name};
            str  = sprintf('%s#', name{:});
            num  = sscanf(str, [prefix,'%d.png#']);
            [~, index] = sort(num);
            
            for t=2:self.n_samples
                im = imread([path,files(index(t)).name]);
                self.stimulus(:,:,t) = logical(im(:,:,1));
                waitbar(t/self.n_samples,wb)
            end
            self.stimulus = reshape(self.stimulus,...
                self.r_image^2,self.n_samples);
            close(wb)
        end
        
        function create_timecourses(self,varargin)
            % creates predicted timecourses based on the effective stimulus
            % and a range of isotropic receptive fields for a grid of
            % location (x,y) and size parameters.
            %
            % optional inputs are
            %  - min_XY     : lower bound of x & y location   (default = -10.0)
            %  - max_XY     : upper bound of x & y location   (default =  10.0)
            %  - number_XY  : steps from lower to upper bound (default =  30.0)
            %  - min_size   : lower bound of RF size          (default =   0.2)
            %  - max_size   : upper bound of RF size          (default =   7.0)
            %  - number_size: steps from lower to upper bound (default =  10.0)
            
            wb = waitbar(0,'creating timecourse...',...
                'Name','pRF mapping tool');
            
            p = inputParser;
            addOptional(p,'number_XY',30);
            addOptional(p,'min_XY',10);
            addOptional(p,'max_XY',10);
            addOptional(p,'number_size',10);
            addOptional(p,'min_size',0.2);
            addOptional(p,'max_size',7);
            p.parse(varargin{:});
            
            n_xy = p.Results.number_XY;
            min_xy = p.Results.min_XY;
            max_xy = p.Results.max_XY;
            n_size = p.Results.number_size;
            min_size = p.Results.min_size;
            max_size = p.Results.max_size;
            self.n_points = n_xy^2*n_size;
            
            [X,Y] = meshgrid(linspace(min_xy,...
                max_xy,self.r_image));
            X = reshape(X,self.r_image^2,1);
            Y = -reshape(Y,self.r_image^2,1);
            
            i = (0:self.n_points-1)';
            self.idx = [floor(i/(n_xy*n_size))+1,...
                mod(floor(i/(n_size)),n_xy)+1,...
                mod(i,n_size)+1];
            self.XY = linspace(min_xy,max_xy,n_xy);
            self.Sigma = linspace(min_size,max_size,n_size);
            
            W = single(zeros(self.n_points,self.r_image^2));
            for j=1:self.n_points
                x = self.XY(self.idx(j,1));
                y = self.XY(self.idx(j,2));
                sigma = self.Sigma(self.idx(j,3))*abs(x+y*1i);
                W(j,:) = self.gauss(x,y,sigma,X,Y)';
                waitbar(j/self.n_points*.9,wb);
            end
            
            tc = W * self.stimulus;
            waitbar(1,wb);
            self.tc_fft = fft(tc');
            close(wb)
        end
        
        function results = mapping(self,data,varargin)
            % identifies the best fitting timecourse for each voxel and
            % returns a structure with the following fields 
            %  - R: model fit per voxel (correlation between empirically 
            %       observed BOLD signal and model timecourse)
            %  - X 
            %  - Y
            %  - Sigma 
            %  - Eccentricity
            %  - Polar_Angle
            %
            % each field is 3-dimensional corresponding to the volumetric
            % dimensions of the data.
            %
            % required inputs are
            %  - data      : a 4-dimensional matrix of empirically observed 
            %                BOLD timecourses. Columns correspond to time 
            %                (volumes/TRs).
            % optional inputs are
            %  - threshold : minimum voxel intensity (default = 100.0)

            
            wb = waitbar(0,'mapping population receptive fields...',...
                'Name','pRF mapping tool');
            
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
            results.Sigma = NaN();
            
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
                        results.X(v) = self.XY(self.idx(j,1));
                        results.Y(v) = self.XY(self.idx(j,2));
                        results.Sigma(v) = self.Sigma(self.idx(j,3));
                    end
                    waitbar(v/numXYZ,wb)
                end
            else
                for v=1:numXYZ
                    if mean_signal(v)>threshold
                        hrf_fft = fft(repmat([self.hrf(:,v);...
                            zeros(self.n_samples-self.l_hrf,1)],...
                            [1,self.n_points]));
                        tc = zscore(ifft(self.tc_fft.*hrf_fft))';
                        mag_tc = sqrt(sum(tc.^2,2));
                        
                        CS = (tc*data(:,v))./...
                            (mag_tc*mag_d(v));
                        id = isinf(CS) | isnan(CS);
                        CS(id) = 0;
                        [results.R(v),j] = max(CS);
                        results.X(v) = self.XY(self.idx(j,1));
                        results.Y(v) = self.XY(self.idx(j,2));
                        results.Sigma(v) = self.Sigma(self.idx(j,3));
                    end
                    waitbar(v/numXYZ,wb)
                end
            end
            results.R = reshape(results.R,numX,numY,numZ);
            results.X = reshape(results.X,numX,numY,numZ);
            results.X = reshape(results.Y,numX,numY,numZ);
            results.Sigma = reshape(results.Sigma,numX,numY,numZ);
            results.Eccentricity = abs(results.X+results.Y*1i);
            results.Polar_Angle = angle(results.X+results.Y*1i);
            close(wb)
        end
    end
end