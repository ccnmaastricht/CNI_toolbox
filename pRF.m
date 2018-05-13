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
    end
    
    properties (Access = public)
        hrf
        stimulus
        tc_fft
    end
    
    methods (Access = public)
        
        function self = pRF(scan_params,varargin)
            % constructor
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
        
        function load_stimulus(self)
            % bla bla
            %
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
            % bla bla
            %
            
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
            % bla bla
            %
            
            wb = waitbar(0,'mapping population receptive fields...',...
                'Name','pRF mapping tool');
            
            p = inputParser;
            addRequired(p,'data',@isnumeric);
            addOptional(p,'stimulus',[]);
            addOptional(p,'threshold',100);
            p.parse(data,varargin{:});
            
            data = p.Results.data;
            self.stimulus = p.Results.stimulus;
            if ndims(self.stimulus)==3
                self.r_image = size(self.stimulus,1);
                self.stimulus = reshape(self.stimulus,...
                    self.r_image^2,self.n_samples);
            else
                self.r_image = sqrt(size(self.stimulus,1));
            end
            
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