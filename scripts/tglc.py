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
        if at_top == at_bottom == 0:
            return
        
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


def assemble_sections(input_file_template, output_file,
                      section_list, ms_only=False, edge_thickness=0):
    """Assemble core sections into an entire core.

    section_list is a list of 4-tuples. Each 4-tuple has the
    following form:

    (name [str], section_length [int], empty_bottom [int], empty_top [int],
     top_space [int, optional])

    "name" is a string label for the section, used to find the data
    file. section_length is the length of the actual core section,
    which may be shorter than the length of magnetometer track measured.
    empty_bottom and empty_top specify the lengths of the empty track
    sections below and above the core. These three integers should add
    up to the total length of measured data in the corresponding file.
    (If they don't, this function will throw an assertion error.)

    top_space may be omitted. If included, it denotes a non-sampled space
    at the top of the core section (e.g. if some material was lost when
    sectioning or u-channelling the core). top_space is added to to the
    depth counter at the top of the core, so it has the effect of offsetting
    all depths from the top of that section downwards by the specified
    amount, increasing the total core length.

    The ordering of the sections in section_list determines the
    order of section assembly. Sections are ordered from top to
    bottom, so the top of the first section in the list will be at 0 cm
    depth.

    A note on the empty_bottom and empty_top parameters: these are
    used to remove measured data that doesn't come from the core itself
    and doesn't correspond to any sampled depth. Such measurements are
    taken when the core section itself is shorter than the measured
    track length. For instance, if the 2G software is told to measure
    110cm of data, and a 100cm core is placed at a 5cm offset from
    the start of the track, the first and last 5cm will be "empty"
    measurements and can be removed from the data using empty_bottom
    and empty_top parameters.

    The sections removed using empty_bottom and empty_top are really
    removed from the depth stack. In contrast, data erased to avoid edge
    effects in the top/bottom few cm still leaves a gap with an
    associated depth: it's a missing measurement at a valid depth,
    whereas empty_bottom / empty_top removes measurements which aren't
    even associated with a valid depth.

    There is a slight fencepost complication in the way depths are
    added up. On, say, a 100 cm core, measurements are taken at each
    point from 0 cm to 100 cm. The number of measurements is thus
    one more than the number of centimetres in the core. The first
    and last measurements don't have a unique depth: the bottom-most
    measurement of a core is at the same depth as the topmost measurement
    from the core below it. Thus, if we do an "edge-effect"  removal
    of four measurements from the adjacent ends of two cores, we're actually
    only blanking out measurements through 2 * 4 - 1 = 7 cm of depth.

    Args:
        input_file_template: a template for the input filename. It should
            contain the string "%s", which will be replaced in turn with
            each "name" field from section_list.
        output_file: name for the the assembled file to be written
        section_list: list of section codes and truncation specifiers
        ms_only: True iff the files contain only magnetic susceptibility data
        edge_thickness: number of measurements to remove at the top and
            bottom of each section to avoid edge effects. This will produce
            depth gaps between sections.

    """

    section_names = [section[0] for section in section_list]
    section_dict = {section[0]: (section[1], section[2], section[3],
                                 section[4] if len(section) == 5 else 0)
                    for section in section_list}

    sections = []
    top = 0
    for section in section_names:
        f = File()
        f.read(input_file_template % section)
        section_length, empty_bottom, empty_top, extra_space = \
            section_dict[section]
        top += extra_space
        f.truncate(empty_bottom,
                   empty_top)  # chop off the empty tray
        thickness = f.get_thickness()  # thickness of the actual mud
        assert thickness == section_length, \
            "specified length %d does not equal actual length %d" % \
            (section_length, thickness)
        f.truncate(edge_thickness, edge_thickness)

        def depth(x):
            return top + thickness - x*100 + empty_bottom

        f.change_depth(depth)
        if ms_only:
            f.chop_fields("Depth\tMS corr\n")
        sections.append(f)
        top += thickness

    composite = File()
    composite.concatenate(sections)
    composite.sort()
    composite.write(output_file)
