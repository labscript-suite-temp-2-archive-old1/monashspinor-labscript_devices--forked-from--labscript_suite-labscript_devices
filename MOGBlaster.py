#####################################################################
#                                                                   #
# MOGblaster.py                                                      #
#                                                                   #
# Copyright 2017, Monash University                                 #
#                                                                   #
# This file is part of labscript_devices, in the labscript suite    #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################
from labscript_devices import labscript_device, BLACS_tab, BLACS_worker, runviewer_parser
from labscript import Device, PseudoclockDevice, Pseudoclock, ClockLine, IntermediateDevice, DigitalQuantity, DigitalOut, DDS
from labscript import config, startupinfo, LabscriptError, set_passed_properties
from labscript_utils.unitconversions import MOGLabsDDSFreqConversion, MOGLabsDDSAmpConversion
from labscript_utils.setup_logging import setup_logging
import numpy as np

# Global variables
BAUD_RATE = 115200
DEFAULT_PORT = 7802
DEFAULT_TIMEOUT = 3
FREQ_BITS = 32
AMPL_BITS = 14
PHASE_BITS = 16
FREQ_MIN = 20.0e6
FREQ_MAX = 400e6

# Define a MOGBlasterPseudoclock that only accepts one child clockline
class MOGBlasterPseudoclock(Pseudoclock):    
    def add_device(self, device):
        if isinstance(device, ClockLine):
            # only allow one child
            if self.child_devices:
                raise LabscriptError('The pseudoclock of the MOGBlaster %s only supports 1 clockline, which is automatically created. Please use the clockline located at %s.clockline' % (self.parent_device.name, self.parent_device.name))
            Pseudoclock.add_device(self, device)
        else:
            raise LabscriptError('You have connected %s to %s (the Pseudoclock of %s), but %s only supports children that are ClockLines. Please connect your device to %s.clockline instead.' % (device.name, self.name, self.parent_device.name, self.name, self.parent_device.name))

@labscript_device
class MOGBlaster(PseudoclockDevice):
    description = 'MOGLabs-XRF421'
    clock_limit = 1.0e6
    clock_resolution = 13.33333333333333333333e-9
    allowed_children = [MOGBlasterPseudoclock]
    
    # TODO: find out what these actually are!
    trigger_delay = 1.0e-6
    wait_delay = trigger_delay
    
    @set_passed_properties()
    def __init__(self, name, addr, port=None,
                 trigger_device=None, trigger_connection=None, **kwargs):
        PseudoclockDevice.__init__(self, name, trigger_device, trigger_connection)
        if addr.startswith('COM') or addr == 'USB':
            if port is not None: addr = 'COM%d'%port
            addr = addr.split(' ', 1)[0]
            addr = '%s,%i' % (addr, BAUD_RATE)
            self.BLACS_connection = addr
            self.is_usb = True
        else:
            if not ':' in addr:
                if port is None: port=DEFAULT_PORT
                addr = '%s:%d' % (addr, port)
            self.BLACS_connection = addr
            self.is_usb = False
        
        # create Pseudoclock and clockline
        self._pseudoclock = MOGBlasterPseudoclock('%s_pseudoclock'%name, self, 'clock') # possibly a better connection name than 'clock'?
        # Create the internal direct output clock_line
        self._clock_line = ClockLine('%s_clock_line' % name, self.pseudoclock, 'internal')
        # Create the internal intermediate device connected to the above clock line
        # This will have the DDSs of the MOGBlaster connected to it
        self._direct_output_device = MOGBlasterDirectOutputs('%s_direct_output_device' % name, self._clock_line)
    
    @property
    def pseudoclock(self):
        return self._pseudoclock
    
    @property
    def direct_outputs(self):
        return self._direct_output_device
    
    def add_device(self, device):
        if not self.child_devices and isinstance(device, Pseudoclock):
            PseudoclockDevice.add_device(self, device)
        elif isinstance(device, Pseudoclock):
            raise LabscriptError('The %s %s automatically creates a Pseudoclock because it only supports one. ' % (self.description, self.name) +
                                 'Instead of instantiating your own Pseudoclock object, please use the internal' +
                                 ' one stored in %s.pseudoclock'%self.name)
        elif isinstance(device, DDS):
            #TODO: Defensive programming: device.name may not exist!
            raise LabscriptError('You have connected %s directly to %s, which is not allowed. You should instead specify the parent_device of %s as %s.direct_outputs' % (device.name, self.name, device.name, self.name))
        else:
            raise LabscriptError('You have connected %s (class %s) to %s, but %s does not support children with that class.' % (device.name, device.__class__, self.name, self.name))
        
        
    def generate_code(self, hdf5_file):                
        # Generate clock and save raw instructions to the h5 file:
        PseudoclockDevice.generate_code(self, hdf5_file)

        # Datatypes for the table of human-readable quantities
        dtypes = [('time', float), ('amp0', float), ('freq0', float), ('phase0', float), ('dds_en0', bool), 
                                   ('amp1', float), ('freq1', float), ('phase1', float), ('dds_en1', bool)]        
        times = self.pseudoclock.times[self._clock_line]
        
        # Create an empty array of these types
        data = np.zeros(len(times), dtype=dtypes)
        data['time'] = times
        for dds in self.direct_outputs.child_devices:
            prefix, connection = dds.connection.split()
            data['freq%s' % connection] = dds.frequency.raw_output
            data['amp%s' % connection] = dds.amplitude.raw_output
            data['phase%s' % connection] = dds.phase.raw_output
            data['dds_en%s' % connection] = dds.gate.raw_output

        group = hdf5_file['devices'].create_group(self.name)
        group.create_dataset('TABLE_DATA', compression=config.compression, data=data)
        
        # Quantise the data and save it to the h5 file:
        quantised_dtypes = [('time', np.int64),
                            ('amp0', np.int16), ('freq0', np.int32), ('phase0', np.int16), ('dds_en0', bool),
                            ('amp1', np.int16), ('freq1', np.int32), ('phase1', np.int16), ('dds_en1', bool)]
        quantised_data = np.zeros(len(times), dtype=quantised_dtypes)
        quantised_data['time'] = np.array(1e6/self.clock_limit*data['time']+0.5)
        for dds in range(2):
            # Adding 0.5 to each so that casting to integer rounds:
            # TODO: bounds checking
            quantised_data['freq%d' % dds]   = np.array((2**FREQ_BITS-1)*data['freq%d' % dds]/FREQ_MAX + 0.5)
            quantised_data['amp%d' % dds]    = np.array((2**AMPL_BITS-1)*data['amp%d' % dds] + 0.5)
            quantised_data['phase%d' % dds]  = np.array((2**PHASE_BITS-1)*data['phase%d' % dds]/360. + 0.5)
            quantised_data['dds_en%d' % dds] = data['dds_en%d' % dds]
        group.create_dataset('QUANTISED_DATA', compression=config.compression, data=quantised_data)

        # When should the MOGBlaster wait for a trigger?
        trigger_times = self.trigger_times
        if trigger_times is not None:
            group['TABLE_DATA'].attrs.create('trigger_times', trigger_times)


class MOGBlasterDirectOutputs(IntermediateDevice):
    description = 'MOGLabs-XRF421 DDS outputs'
    allowed_children = [DDS]
    clock_limit = MOGBlaster.clock_limit
  
    def add_device(self, device):       
        IntermediateDevice.add_device(self, device)
        if isinstance(device, DDS):
            # Check that the user has not specified another digital line as the gate for this DDS, that doesn't make sense.
            # Then instantiate a DigitalQuantity to keep track of gating.
            if device.gate is None:
                device.gate = DigitalQuantity(device.name + '_gate', device, 'gate')
            else:
                raise LabscriptError('You cannot specify a digital gate ' +
                                     'for a DDS connected to %s. '% (self.name) + 
                                     'The digital gate is always internal to the MOGBlaster.')

    def get_default_unit_conversion_classes(self, device):
        """Child devices call this during their __init__ (with themselves
        as the argument) to check if there are certain unit calibration
        classes that they should apply to their outputs, if the user has
        not otherwise specified a calibration class"""
        if device.connection in ['dds 0', 'dds 1']:
            # Default calibration classes for the non-static channels:
            return MOGLabsDDSFreqConversion, MOGLabsDDSAmpConversion, None
        else:
            return None, None, None


from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED  
from blacs.device_base_class import DeviceTab


@BLACS_tab
class MOGBlasterTab(DeviceTab):
    def initialise_GUI(self):
        # Conversions
        self.freq_conv = MOGLabsDDSFreqConversion()
        self.amp_conv = {}
        self.amp_conv[1] = MOGLabsDDSAmpConversion({'channel': 1})
        self.amp_conv[2] = MOGLabsDDSAmpConversion({'channel': 2})

        # Capabilities
        self.base_units =    {'freq': 'Hz',    'amp': 'int',          'phase': 'Degrees'}
        self.base_min =      {'freq': 20.0e6,  'amp': 0,              'phase': 0}
        self.base_max =      {'freq': 170.0e6, 'amp': 2**AMPL_BITS-1, 'phase': 360}
        self.base_step =     {'freq': 1.0e6,   'amp': 1,              'phase': 1}
        self.base_decimals = {'freq': 3,       'amp': 0,              'phase': 3} # TODO: find out what the phase precision is!
        self.num_DDS = 2
        self.num_DO = 8

        # Create DDS Output objects
        dds_prop = {}
        for i in range(self.num_DDS): # 2 is the number of DDS outputs on this device
            dds_prop['dds %d' % i] = {}
            # for subchnl in ['freq', 'amp', 'phase']:
            #     dds_prop['dds %d' % i][subchnl] = {'base_unit': self.base_units[subchnl],
            #                                        'min': self.base_min[subchnl],
            #                                        'max': self.base_max[subchnl],
            #                                        'step': self.base_step[subchnl],
            #                                        'decimals': self.base_decimals[subchnl]
            #                                     }
            dds_prop['dds %d' % i]['gate'] = {}
        for ch in [1, 2]:
            dds_prop['dds %d' % (ch-1)]['freq'] = {'base_unit': 'Hz', 'min': self.freq_conv.freq_min, 'max': self.freq_conv.freq_max, 'step': self.freq_conv.df, 'decimals': 3}
            dds_prop['dds %d' % (ch-1)]['amp'] = {'base_unit': 'int', 'min': self.amp_conv[ch].amp_min, 'max': self.amp_conv[ch].amp_max, 'step': 1, 'decimals': 0}
            dds_prop['dds %d' % (ch-1)]['phase'] = {'base_unit': 'Degrees', 'min': 0, 'max': 360, 'step': 1, 'decimals': 3}

        do_prop = {}
        for i in range(self.num_DO):
            do_prop['flag %d' % i] = {}
                
        # Create the output objects    
        self.create_dds_outputs(dds_prop)
        self.create_digital_outputs(do_prop)        
        # Create widgets for output objects
        dds_widgets, ao_widgets, do_widgets = self.auto_create_widgets()

        # Define the sort function for the digital outputs
        def sort(channel):
            flag = channel.replace('flag ','')
            flag = int(flag)
            return '%02d'%(flag)
        
        # and auto place the widgets in the UI
        self.auto_place_widgets(("DDS Outputs", dds_widgets), ("Flags", do_widgets,sort))

        # Get connection details
        connection_object = self.settings['connection_table'].find_by_name(self.device_name)
        # Store the COM port / IP address to be used
        blacs_connection =  str(connection_object.BLACS_connection)
        if ',' in blacs_connection:
            self.is_usb = True
            self.com_port, baud_rate = blacs_connection.split(',')
            self.baud_rate = int(baud_rate)
            self.ip, self.port = None, None
        elif ':' in blacs_connection:
            self.is_usb = False
            self.ip, port = blacs_connection.split(':')
            self.port = int(port)
            self.com_port, self.baud_rate = None, None
        
        # Create and set the primary worker
        self.create_worker("main_worker", MOGBlasterWorker, {'com_port': self.com_port,
                                                             'baud_rate': self.baud_rate,
                                                             'ip': self.ip,
                                                             'port': self.port,
                                                             'is_usb': self.is_usb,
                                                             'num_DDS': self.num_DDS,
                                                             'num_DO': self.num_DO
                                                             })
        self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_remote_value_check(True)
        self.supports_smart_programming(True)
    
    def get_child_from_connection_table(self, parent_device_name, port):
        # This is a direct output, let's search for it on the internal intermediate device called 
        # MOGBlasterDirectOutputs
        if parent_device_name == self.device_name:
            device = self.connection_table.find_by_name(self.device_name)
            pseudoclock = device.child_list[device.child_list.keys()[0]] # there should always be one (and only one) child, the Pseudoclock
            clockline = pseudoclock.child_list[pseudoclock.child_list.keys()[0]] # there should always be one (and only one) child, the clockline
            direct_outputs = clockline.child_list[clockline.child_list.keys()[0]] # There should only be one child of this clock line, the direct outputs
            # look to see if the port is used by a child of the direct outputs
            return DeviceTab.get_child_from_connection_table(self, direct_outputs.name, port)
        else:
            # else it's a child of a DDS, so we can use the default behaviour to find the device
            return DeviceTab.get_child_from_connection_table(self, parent_device_name, port)
    
    # We override this because the RFBlaster doesn't really support remote_value_checking properly
    # Here we specifically do not program the device (it's slow!) nor do we update the last programmed value to the current
    # front panel state. This is because the remote value returned from the RFBlaster is always the last *manual* values programmed.
    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        self.mode = MODE_TRANSITION_TO_MANUAL
        
        success = yield(self.queue_work(self._primary_worker, 'transition_to_manual'))
        for worker in self._secondary_workers:
            transition_success = yield(self.queue_work(worker, 'transition_to_manual'))
            if not transition_success:
                success = False
                # don't break here, so that as much of the device is returned to normal
        
        # Update the GUI with the final values of the run:
        for channel, value in self._final_values.items():
            if channel in self._AO:
                self._AO[channel].set_value(value, program=False)
            elif channel in self._DO:
                self._DO[channel].set_value(value, program=False)
            elif channel in self._DDS:
                self._DDS[channel].set_value(value, program=False)
        
        if success:
            notify_queue.put([self.device_name, 'success'])
            self.mode = MODE_MANUAL
        else:
            notify_queue.put([self.device_name, 'fail'])
            raise Exception('Could not transition to manual. You must restart this device to continue')
            
    
@BLACS_worker
class MOGBlasterWorker(Worker):
    def init(self, timeout=DEFAULT_TIMEOUT, check=True, debug=True):
        global serial; import serial
        global socket; import socket
        global select; import select
        global h5py; import labscript_utils.h5_lock, h5py
        self.logger = setup_logging(self.device_name)
        self.logger.info('init: Started logging')
        self.smart_cache = {'QUANTISED_DATA': '', 'TABLE_DATA': ''}
        self.timeout = timeout # How long do we wait until we assume that the MOGBlaster is dead? (in seconds)
        self.check = check
        self._DEBUG = debug
        self.freq_conv = MOGLabsDDSFreqConversion()
        self.amp_conv = {}
        self.amp_conv[1] = MOGLabsDDSAmpConversion({'channel': 1})
        self.amp_conv[2] = MOGLabsDDSAmpConversion({'channel': 2})
        self.first_table = False

        # See if the RFBlaster connects
        self.reconnect(self.timeout, self.check)
        
        # TODO: Find out what this does?
        self._last_program_manual_values = {}

        # Keep track of whether or not to clear the tables of both channels
        clear_tables = False 

        # Get each channel into a well defined state at startup, ready for program_manual and and transition_to_buffered
        for ch in [1, 2]:
            self.cmd('MODE,%i,TSB' % ch)             # set into table mode
            # self.cmd('ON,%i,POW' % ch)             # turn on the amplifiers
            # self.cmd('TABLE,LENGTH,%i,2' % ch)       # just use three dummy instructions until we get a fresh table
            # self.cmd('TABLE,CLEAR,%i' % ch)
            table_length = self.ask('TABLE,LENGTH,%i' % ch)
            table_length = int(table_length.split()[0])
            if table_length < 3:
                clear_tables = True
            self.cmd('TABLE,RESTART,%i,OFF' % ch)    # disable automatic table restart
        # Do we need to clear both tables (a reprogam of these will occur upon check_remote_values)
        if clear_tables:
            self.logger.info('Fewer than 2 table entries. Clearing both tables')
            for ch in [1, 2]:
                self.cmd('TABLE,CLEAR,%i' % ch)
        # self.program_manual(startup_values, update_output=False)    # ensure valid initial table entries
        self.cmd('TABLE,REARM,1,OFF')               # disable automatic table rearm on CH1
        self.cmd('TABLE,REARM,2,OFF')                # enable automatic table rearm on CH2
        self.cmd('TABLE,TRIGSYNC,1')                # trigger sync (CH2 also starts on falling edge of pin3 of DB15)
        self.cmd('TABLE,SYNC,1')                    # enable synchronous table mode of DDS channels (CH1 master)

    #================ GENERIC COMMUNICATIONS METHODS ================
    def reconnect(self, timeout=DEFAULT_TIMEOUT, check=True):
        "Reestablish connection with unit"
        if hasattr(self, 'dev'): self.dev.close()
        if self.is_usb:
            self.dev = serial.Serial(self.com_port, self.baud_rate, bytesize=8, parity='N', stopbits=1, timeout=timeout, writeTimeout=0)
        else:
            self.dev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dev.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.dev.settimeout(timeout)
            self.dev.connect((self.ip, self.port))
        # check the connection?
        if check:
            try:
                self.info = self.ask('info')
                self.serial = self.ask('get,serial')
            except Exception as E:
                print '!', E
                raise Exception('Device did not respond to query.\n' + str(E))

    def cmd(self, cmd):
        "Send the specified command, and check the response is OK"
        self.flush()
        self.send(cmd)
        resp = self.recv()
        if resp.startswith('OK'):
            return resp
        else:
            self.logger.error(cmd)
            raise Exception(resp)
        
    def ask(self, cmd):
        "Send followed by receive"
        # check if there's any response waiting on the line
        self.flush()
        self.send(cmd)
        resp = self.recv().strip()
        if resp.startswith('ERR:'):
            raise Exception(resp[4:].strip())
        return resp
        
    def ask_bin(self, cmd):
        "Send a request which returns a binary response"
        self.send(cmd)
        head = self.recv_raw(4)
        print repr(head)
        # is it an error message?
        if head == 'ERR:': raise RuntimeError(head+self.recv())
        datalen = struct.unpack('<L',head)[0]
        data = self.recv_raw(datalen)
        if len(data) != datalen: raise RuntimeError('Binary response block has incorrect length')
        return data
    
    def send(self,cmd):
        "Send command, appending newline if not present"
        if not cmd.endswith('\r\n'): cmd += '\r\n'
        self.send_raw(cmd)
    
    def has_data(self, timeout=0):
        if self.is_usb:
            if self.dev.inWaiting(): return True
            if timeout == 0: return False
            time.sleep(timeout)
            return self.dev.inWaiting > 0
        else:
            return len(select.select([self.dev],[],[],timeout)[0]) > 0
        
    def flush(self, buffer=256):
        while self.has_data():
            dat = self.recv(buffer)
            if self._DEBUG: print 'FLUSHED', repr(dat)
    
    def recv(self, buffer=256):
        "A somewhat robust multi-packet receive call"
        if self.is_usb:
            data = self.dev.readline(buffer)
            if len(data):
                t0 = self.dev.timeout
                self.dev.timeout = 0 if data.endswith('\r\n') else 0.1
                while True:
                    segment = self.dev.readline(buffer)
                    if len(segment) == 0: break
                    data += segment
                self.dev.timeout = t0
            if len(data) == 0: raise RuntimeError('timed out')
        else:
            data = self.dev.recv(buffer)
            timeout = 0 if data.endswith('\r\n') else 0.1
            while self.has_data(timeout):
                try:
                    segment = self.dev.recv(buffer)
                except IOError:
                    if len(data): break
                    raise
                data += segment
        if self._DEBUG: print '<<',len(data),repr(data)
        return data
    
    def send_raw(self,cmd):
        "Send, without appending newline"
        if self._DEBUG and len(cmd) < 256: print '>>',repr(cmd)
        if self.is_usb:
            return self.dev.write(cmd)
        else:
            return self.dev.send(cmd)
    
    def recv_raw(self,size):
        "Receive exactly 'size' bytes"
        buffer = ''
        while size > 0:
            if self.is_usb:
                chunk = self.dev.read(size)
            else:
                chunk = self.dev.recv(size)
            buffer += chunk
            size -= len(chunk)
        if self._DEBUG:
            print '<< RECV_RAW got', len(buffer)
            print '<<', repr(buffer)
        return buffer

    def close(self):
        if hasattr(self, 'dev'): self.dev.close()

    #================ SPECIFIC COMMUNICATIONS METHODS ================
    def parse_entry(self, response, ch=1):
        val_strings = [s.strip() for s in response.split(',')]
        vals = {}
        keys = ['freq', 'amp', 'phase']
        for k, x in zip(keys, val_strings[:3]):
            vals[k] = float(x)
        vals['freq'] = self.freq_conv.MHz_to_base(vals['freq'])
        vals['amp'] = self.amp_conv[ch].dBm_to_base(vals['amp'])
        vals['gate'] = not any([x.lower().endswith('off') for x in val_strings[4:]])
        wait = any([x.lower().startswith('trig') for x in val_strings[4:]])
        return vals

    def parse_hex_entry(self, response):
        val_strings = [s.strip() for s in response.split(',')]
        vals = {}
        keys = ['freq', 'amp', 'phase']
        for k, x in zip(keys, val_strings[:3]):
            vals[k] = eval(x)
        vals['freq'] = self.freq_conv.int_to_base(vals['freq'])
        vals['gate'] = not any([x.lower().endswith('off') for x in val_strings[4:]])
        wait = any([x.lower().startswith('trig') for x in val_strings[4:]])
        return vals

    def get_status(self, channel):
        response = self.ask('STATUS,%i', channel)
        status = dict([[y.strip() for y in x.split(':')] for x in response.split(',')])
        # status['AMP'] = status.pop('POW')
        return status

    #================ METHODS REQUIRED BY BLACS ======================
    def check_remote_values(self):
        self.logger.info('Checking remote values.')
        try:
            results = {}
            for ch in [1, 2]:
                # Check the length of the programmed table
                table_length = self.ask('TABLE,LENGTH,%i' % ch)
                table_length = int(table_length.split()[0])
                if table_length:
                    response = self.ask('TABLE,HEXENTRY,%i,1' % ch)
                    vals = self.parse_hex_entry(response)
                else:
                    self.logger.info('No table in memory. Returning default values.')
                    vals = {}
                    vals['freq'] = 80.0e6
                    vals['amp'] = 0.
                    vals['phase'] = 0.
                    vals['gate'] = False
                results['dds %d' % (ch-1)] = vals
        except socket.timeout:
            raise Exception('Failed to check remote values. Timed out.')
        for i in range(self.num_DO):
            results['flag %d' % i] = 0
        return results

    def program_manual(self, front_panel_values, update_output=True):
        for ch in [1, 2]:
            # Get a dictionary of front panel values for this channel
            vals = front_panel_values['dds %d' % (ch-1)]
            if 'dds %d' % (ch-1) in self._last_program_manual_values:
                last_vals = self._last_program_manual_values['dds %d' % (ch-1)]
            else:
                last_vals = {}

            if vals == last_vals:
                pass
            
            # Check the length of the programmed table
            table_length = self.ask('TABLE,LENGTH,%i' % ch)
            table_length = int(table_length.split()[0])

            # If fewer than three instructions (no buffered output programmed), clear the table
            if table_length < 3:
                self.logger.info('No buffered output. Clearing table on channel %i.' % ch)
                self.cmd('TABLE,CLEAR,%i' % ch)

            # Create two short instructions with the new front_panel_values, wait on the second
            command_string = 'TABLE,ENTRY,%i,line,%f,0x%x,%f,1' % (ch, self.freq_conv.MHz_from_base(vals['freq']), vals['amp'], vals['phase'])
            if not vals['gate']:
                command_string += ',OFF'
            self.cmd(command_string.replace('line', '1'))
            command_string += ',TRIG'     # Make the second instruction wait on a falling edge of the DB15/SEQ input (pin 3)
            self.cmd(command_string.replace('line', '2'))
            
            # If no buffered output programmed, program a third (dummy) instruction
            if table_length < 3:
                self.cmd(command_string.replace('line', '3').replace(',TRIG', ''))
                table_length = 3

            # Update the declared table length
            self.cmd('TABLE,LENGTH,%i,%i' % (ch, table_length))

        if update_output:
            # Stop the table and restart it to load the new values
            self.cmd('TABLE,STOP,1')
            self.cmd('TABLE,ARM,2')
            self.cmd('TABLE,START,1')

        self._last_program_manual_values = front_panel_values
        return self.check_remote_values()
        
    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.h5file = h5file

        # Store the initial values in case we have to abort and restore them:
        self.initial_values = initial_values

        # Store the final values to for use during transition_to_static:
        self.final_values = {}

        # Get the table data from file
        table_data = None
        quantised_table_data = None
        with h5py.File(h5file) as hdf5_file:
            group = hdf5_file['devices'][device_name]
            if 'TABLE_DATA' in group:
                table_data = group['TABLE_DATA'][:]
                trigger_times = group['TABLE_DATA'].attrs.get('trigger_times', default=[])
                trigger_ix = 0
            if 'QUANTISED_DATA' in group:
                quantised_data = group['QUANTISED_DATA'][:]

        # Use the unquantised data for programming
        data = table_data
        # data = quantised_data

        # Now program the buffered outputs:
        if data is not None:
            # Get the old data from the smart_cache for comparison (not implemented!)
            oldtable = self.smart_cache['TABLE_DATA']
            # Check the length of the programmed table
            for ch in [1, 2]:
                self.logger.info('Checking length of table for channel %i' % ch)
                table_length = self.ask('TABLE,LENGTH,%i' % ch)
                table_length = int(table_length.split()[0])
                # If necessary, change the number of table entries
                if table_length != len(data) + 2:
                    self.logger.info('Channel %i table has %i entries, but we need %i. Reshaping table.' % (ch, table_length, len(data)+2))
                    self.cmd('TABLE,LENGTH,%i,%i' % (ch, len(data)+2))

            # Calculate the instruction durations
            durations = np.diff(data['time'])   # TODO: Store the durations instead of absolute times in TABLE_DATA

            # Program each row of the table
            for i, line in enumerate(data):
                # For each row, program each channel
                for ch in [1, 2]:
                    new_vals = {}
                    old_vals = {}
                    for key in ['freq', 'amp', 'phase', 'dds_en']:
                        new_vals[key] = line['%s%d' % (key, ch-1)]
                        if i < len(oldtable):
                           old_vals[key] = oldtable[i]['%s%d' % (key, ch-1)]
                    if fresh or i >= len(oldtable) or line['time'] != oldtable[i]['time'] or old_vals != new_vals:
                        ix = i+2    # First two instructions are for static updates
                        ix += 1     # mogrf indexing begins at 1
                        if i < len(data)-1:
                            duration = durations[i]*1e6
                        else:
                            duration = 1
                        inst = 'TABLE,ENTRY,%i,%i,%f,0x%x,%f,%f' % (ch, ix, self.freq_conv.MHz_from_base(new_vals['freq']), int(new_vals['amp']), new_vals['phase'], duration)
                        # Should the channel be disabled during this intruction?
                        if not new_vals['dds_en']:
                            inst += ',OFF'
                        # Is this a wait instruction?
                        if False: # and trigger_ix < len(trigger_times) and line['time'] == trigger_times[trigger_ix]:
                            inst += ',TRIG'
                            trigger_ix += 1
                        self.logger.info('Programming table entry %i of channel %i' % (ix, ch))
                        self.logger.info(inst)
                        self.cmd(inst)
            for ch in [1]:
                # Stop the table and restart it to load the new values
                self.cmd('TABLE,STOP,%i' % ch)
                self.cmd('TABLE,ARM,2')
                self.cmd('TABLE,START,%i' % ch)
            for i in range(self.num_DDS):
                # Find the final value from the human-readable part of the h5 file to use for
                # the front panel values at the end
                self.final_values['dds %d' % i] = {'freq': data["freq%d" % i][-1],
                                                   'amp': data["amp%d" % i][-1],
                                                   'phase': data["phase%d" % i][-1],
                                                   'gate': data["dds_en%d" % i][-1]
                                                  }
            for i in range(self.num_DO):
                self.final_values['flag %d' % i] = 0

            # Store the table for future smart programming comparisons:
            try:
                self.smart_cache['TABLE_DATA'][:len(data)] = data
                self.logger.info('Stored new table as subset of old table')
            except: # new table is longer than old table
                self.smart_cache['TABLE_DATA'] = data
                self.logger.info('New table is longer than old table and has replaced it.')
        return self.final_values
                 
    def abort_transition_to_buffered(self):
        return True
    
    def abort_buffered(self):
        return True
     
    def transition_to_manual(self):
        # for ch in [1, 2]:
        #     self.cmd('TABLE,ARM,%i,1' % ch)
        return True
        
    def shutdown(self):
        self.close()
        pass