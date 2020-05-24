classdef online_processor < handle
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%                               LICENSE                             %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Copyright 2020 Mario Senden
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
    % Class for real time processing of fMRI data and stimulus respone.
    %
    % obj = online_processor(n_channels) creates an instance of the class.
    % n_channels determines the number independent channels (voxels,
    % pixels etc.) to which processing is applied
    %
    % optional inputs are
    %   - sampling_rate: sampling rate of fMRI acquisition (TR) in seconds.
    %     Standard value is 2.
    %   - l_kernel: length of hemodynic convolution kernel in seconds.
    %     Standard value is 34.
    %
    % this class has the following functions
    %
    %   - x_conv = online_processor.convolve(self,x): perform one step of
    %              real-time convolution.
    %   - x_next = online_processor.update(self,x): perform one step of
    %              real-time z-transformation
    %   - online_processor.reset: reset all variables tracking the signal
    properties (Access = private)
        % functions
        two_gamma        % two gamma hrf function
        
        % parameters
        sampling_rate    % sampling rate
        l_kernel         % length of kernel
        n_channels       % number of channels
        
        % variables
        step             % internal step counter
        mean             % running mean
        previous_mean    % mean at one prior step
        M2               % helper variable for calculating sigma
        sigma            % running standard deviation
        kernel           % convolution kernel
        kernel_fft       % Fourier transformed kernel
        conv_x           % running convolution
    end
    methods (Access = public)
        
        function self = online_processor(n_channels,varargin)
            
            % constructor
            p = inputParser;
            addRequired(p,'n_channels',@isnumeric);
            addOptional(p,'sampling_rate',2);
            addOptional(p,'l_kernel',34);
            p.parse(n_channels,varargin{:});
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            
            self.n_channels = p.Results.n_channels;
            self.sampling_rate = p.Results.sampling_rate;
            self.l_kernel = p.Results.l_kernel;
            
            self.step = 1;
            self.mean = zeros(1,self.n_channels);
            self.previous_mean = zeros(1,self.n_channels);
            self.M2 = ones(1,self.n_channels);
            self.sigma = zeros(1,self.n_channels);
            self.kernel = self.two_gamma(0:self.sampling_rate:...
                self.l_kernel)';
            self.kernel_fft = fft(self.kernel);
            self.conv_x = zeros(numel(self.kernel),self.n_channels);
            
        end
        
        function x_conv = convolve(self,x)
            rows = numel(self.kernel) - 1;
            x_fft = fft([x;zeros(rows,self.n_channels)]);
            self.conv_x = [self.conv_x;zeros(1,self.n_channels)];
            self.conv_x(self.step:self.step+rows,:) = ...
                self.conv_x(self.step:self.step+rows,:) + ...
                ifft(x_fft .* self.kernel_fft);
            x_conv = self.conv_x(self.step,:);
        end
        
        function x_next = update(self,x)
            self.update_mean(x);
            self.update_sigma(x);
            x_next = (x - self.mean) ./ self.sigma;
            self.step = self.step + 1;
        end
        
        function reset(self)
            self.step = 1;
            self.mean = zeros(1,self.n_channels);
            self.previous_mean = zeros(1,self.n_channels);
            self.M2 = ones(1,self.n_channels);
            self.sigma = zeros(1,self.n_channels);
            self.kernel = self.two_gamma(0:self.sampling_rate:...
                self.l_kernel)';
            self.kernel_fft = fft(self.kernel);
            self.conv_x = zeros(numel(self.kernel),self.n_channels);
        end
    end
    
    methods (Access = private)
        function update_mean(self,x)
            self.previous_mean = self.mean;
            self.mean = self.mean + (x - self.mean) ./ self.step;
        end
        
        function update_sigma(self,x)
            self.M2 = self.M2 + (x - self.previous_mean) .* (x - self.mean);
            if self.step ==1
                self.sigma = sqrt(self.M2 ./ self.step);
            else
                self.sigma = sqrt(self.M2 ./ (self.step - 1));
            end
        end
    end
    
end