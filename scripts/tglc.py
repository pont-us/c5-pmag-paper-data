#!/usr/bin/env python

"""Classes to read, manipulate, and write 2G magnetometer data files.

The main focus is simple editing procedures on files, rather than data
analysis. This module can be used, for example, to remove measurements
affected by edge effects at the ends of a core; to rewrite depths
for measurements; to combine several data files into one; and of course
to load and save files in the 2G format. It is suitable for use with
both the CPython and Jython interpreters.
"""

from math import sqrt


class Header:

    def __init__(self, line):
        self.line = line.strip()
        self.field_names = self.line.split('\t')
        self.nfields = len(self.field_names)
        self.fields = {}
        for i in range(0, len(self.field_names)):
            self.fields[self.field_names[i]] = i

    def to_string(self):
        return self.line

    def has_field(self, field_name):
        return (field_name in self.fields)

    def add_field(self, field_name):
        self.fields[field_name] = self.nfields
        self.nfields += 1


class Line:

    def __init__(self, header, line):
        self.header = header
        self.parts = line.strip().split('\t')

    def get(self, field):
        return self.parts[self.header.fields[field]]

    def get_id(self):
        return self.get("Sample ID")

    def getfloat(self, field):
        return float(self.parts[self.header.fields[field]])

    def set(self, field, value):
        self.parts[self.header.fields[field]] = value

    def change(self, field, fn):
        index = self.header.fields[field]
        self.parts[index] = fn(self.parts[index])

    def to_string(self):
        return '\t'.join(self.parts)

    def get_depth(self):
        return int(float(self.get('Depth')) * 100)

    def get_moment_Gcm3(self):
        """Return raw moment in emu (gauss * cm^3)"""
        return sqrt(self.getfloat("X mean")**2 +
                    self.getfloat("Y mean")**2 +
                    self.getfloat("Z mean")**2)

    def flip(self):
        self.change('Y corr', lambda x: str(-float(x)))
        self.change('Z corr', lambda x: str(-float(x)))

    def add_value(self, value):
        self.parts.append(value)


class File:

    def __init__(self):
        self.lines = []

    def read(self, filename):
        with open(filename, "U") as fh:
            self.header = Header(fh.readline())
            for line in fh:
                line = Line(self.header, line)
                if len(line.parts)>1:
                    self.lines.append(line)
        if not self.header.has_field("Depth"):
            self.fake_depth()
        self.update_depths()
        #if self.header.has_field("Depth"): self.update_depths()

    def write(self, filename):
        with open(filename, 'w') as fh:
            fh.write(self.header.to_string() + '\n')
            for line in self.lines:
                fh.write(line.to_string() + '\n')

    def fake_depth(self):
        self.header.add_field("Depth")
        depth = 0
        for line in self.lines:
            line.add_value(str(float(depth)/100.))
            depth += 1

    def change(self, field, fn):
        for line in self.lines:
            line.change(field, fn)

    def update_depths(self):
        self.max_depth = 0
        self.min_depth = 1e99
        for line in self.lines:
            depth = line.get_depth()
            if depth > self.max_depth: self.max_depth = depth
            if depth < self.min_depth: self.min_depth = depth

    def get_max_depth(self):
        return self.max_depth

    def get_thickness(self):
        return self.max_depth - self.min_depth

    def change_depth(self, fn):
        def rewrite(string_val):
            float_in = float(string_val)
            float_out = fn(float_in)
            string_out = '%.0f' % float_out
            return string_out
        self.change('Depth', rewrite)

    def truncate(self, at_top, at_bottom):
        new_lines = []
        limit0 = self.min_depth + at_top
        limit1 = self.max_depth - at_bottom
        for line in self.lines:
            depth = line.get_depth()
            if depth >= limit0 and depth <= limit1:
                new_lines.append(line)
        self.lines = new_lines
        self.update_depths()

    def set_depths(self, start=0, increment = 1):
        depth = start
        for line in self.lines:
            line.set('Depth', str(depth))
            depth += increment

    def concatenate(self, files):
        self.header = files[0].header
        for f in files:
            self.lines.extend(f.lines)

    def chop_fields(self, field_def):
        new_header = Header(field_def)
        for line in self.lines:
            new_parts = []
            for field in new_header.field_names:
                new_parts.append(line.get(field))
            line.parts = new_parts
            line.header = new_header
        self.header = new_header

    def sort(self):
        self.lines.sort(key = lambda x: x.get_depth())

    def flip(self):
        for line in self.lines:
            line.flip()

    def extract_step_intensity(self, step):
        data = []
        for line in self.lines:
            if line.get('AF Z') == step:
                data.append([float(line.get("Depth")),
                             float(line.get("Intensity"))])
        return data
