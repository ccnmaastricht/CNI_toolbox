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
    %   - direction = PEA.get_direction(direction);
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
            addpath(pwd)
            
            self.is = 'PEA fitting tool';
            
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
            % This can be either numeric (-1,1) or in form of a string 
            % ('cw','ccw') for clockwise and counterclockwise rotation,
            % respectively. Contracting rings are also consider to move
            % clockwise (-1) while expanding rings are considered 
            % to move counterclockwise (1).
            if isnumeric(direction)
                self.direction = direction;
            else
                if strcmp(direction,'cw')
                    self.direction = 1;
                else
                    self.direction = -1;
                end
                
            end
        end
        
        function results = fitting(self,data)
            % identifies phase and amplitude at stimulation frequency for each voxel and
            % returns a structure with the following fields
            %  - phase
            %  - amplitude
            %
            % each field is 3-dimensional corresponding to the volumetric dimensions of the data.
            %
            % required inputs are
            %  - data     : a 4-dimensional matrix of empirically observed
            %                BOLD timecourses. Columns correspond to time
            %                (volumes).
            
            text = 'performing phase-encoding analysis...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            F = exp(self.direction*2i*pi*self.f_stim*(self.t-self.delay));
            size(F)
            data = zscore(reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total));
            
            results.phase = zeros(self.n_total,1);
            results.amplitude = zeros(self.n_total,1);
            
            for v=1:self.n_total
                C = F * data(:,v);
                results.phase(v) = angle(C);
                results.amplitude(v) = abs(C);
                waitbar(v/self.n_total,wb)
            end
            
            results.phase = reshape(results.phase,...
                self.n_rows,self.n_cols,self.n_slices);
            results.amplitude = reshape(results.amplitude,...
                self.n_rows,self.n_cols,self.n_slices);
            
            close(wb)
            fprintf('done\n');
        end
        
    end
    
end