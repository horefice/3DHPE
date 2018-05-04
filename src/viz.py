import numpy as np
import torch
import visdom
import time

class Viz():
    def __init__(self):
        self.viz = visdom.Visdom(port=8099)

        # self.viz.close(None) #Close all previously
        # self.create_notepad()

    def create_plot(self, xlabel, ylabel, title, opts_dict):
        options = dict(xlabel=xlabel,
                ylabel=ylabel,
                title=title + time.strftime("(%d.%m @ %H:%M:%S)"))
        options.update(opts_dict)
        
        return self.viz.line(
            X=torch.zeros(1).cpu(),
            Y=torch.zeros(1).cpu(),
            opts=options)

    def update_plot(self, x, y, window, type_upd):
        self.viz.line(
            X=np.array([x]),
            Y=np.array([y]),
            win=window,
            update=type_upd)

    def create_notepad(self):
        txt = "This is a notepad.<br><br>"
        cb_txt_window = self.viz.text(txt)

        def type_callback(event):
            if event['event_type'] == 'KeyPress':
                curr_txt = event['pane_data']['content']
                if event['key'] == 'Enter':
                    curr_txt += '<br>'
                elif event['key'] == 'Backspace':
                    curr_txt = curr_txt[:-1]
                elif event['key'] == 'Delete':
                    curr_txt = txt
                elif len(event['key']) == 1:
                    curr_txt += event['key']
                self.viz.text(curr_txt, win=cb_txt_window)

        self.viz.register_event_handler(type_callback, cb_txt_window)

