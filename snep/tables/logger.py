from logging import Handler
import time

class LoggingTableHandler(Handler):
    def __init__(self, table):
        """
        Initialize the handler.
        """
        Handler.__init__(self)
        self.table = table
    
    def emit(self, record):
        """
        Do whatever it takes to actually log the specified logging record.
        """
        record.message = record.getMessage()
        record.exc_text = self.handleExceptionInfo(record)
        t = time.strftime("%Y-%m-%d %H:%M:%S", record.created)
        
        r = self.table.row
        r['time'] = "{0},{1:03d}".format(t, record.msecs)
        r['exception'] = record.exc_text
        r['message'] = record.message
        r.append()
        
        self.flush()
        return
    
    def flush(self):
        """
        Ensure all logging output has been flushed.
        """
        self.table.flush()
    
    def handleExceptionInfo(self, record):
        import cStringIO, traceback
        
        s = ''
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                ei = record.exc_info
                sio = cStringIO.StringIO()
                traceback.print_exception(ei[0], ei[1], ei[2], None, sio)
                s = sio.getvalue()
                sio.close()
        return s
