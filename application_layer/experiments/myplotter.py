#!/usr/bin/env python3
# coding: utf-8
#This library include independent functions to plot the parsed output
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as mgr
plt.rcParams['pdf.fonttype'] = 42
font_dirs = ['./latin-modern-roman', ]
font_files = mgr.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    mgr.fontManager.addfont(font_file)
#font_list = mgr.createFontList(font_files)
#mgr.fontManager.ttflist.extend(font_list)
plt.rcParams['font.family'] = 'Latin Modern Roman'

fontsize=40
figsize = (15, 8)

def plot_lines(X, Y, sys_legends, linestyles, xlabel, ylabel, output_file):
    #This function plot two variables (X,Y) for multiple systems (i.e., competitors)
    #The first 3 inputs should be lists of the same size
    assert len(X) == len(Y) == len(sys_legends)
    linestyles = linestyles[:len(X)]			#trim the unneeded styles
    figs = []
    for x,y,legend,ls in zip(X,Y, sys_legends, linestyles):
        assert len(x) == len(y)
        fig, = plt.plot(x, y, linewidth=5, label=legend, linestyle=ls)
        figs.append(fig)
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95, left=0.13)
    plt.legend(handles=figs, fontsize=fontsize)
    plt.ylabel(ylabel ,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.savefig(f'{output_file}.pdf', bbox_inches = "tight")
    plt.gcf().clear()

def plot_bars(Y, sys_legends, xtick_labels, hatches, xlabel, ylabel, output_file, text=None, stack=None):
    #This function plots bars defined in Y (which is list of lists) for multiple systems (i.e., competitors)
    #sys_legends define the systems' names and xtick_labels define the labels on the x-axis
    #stack (if not None) specifies which bars should be stacked on one another; it should be a list which tells if this bar should be stacked on another one
    assert len(Y) <= len(sys_legends) and len(Y[0]) == len(xtick_labels)
    if stack is not None:
        assert len(Y)==len(stack)
    else:
        stack = [-1] * len(Y)
    ind = np.arange(len(xtick_labels))
    #next, we calculate the positions of the bars (with align = 'left' always)
    start = -stack.count(-1)/2.0
    pos = list(np.arange(start,start*-1,1))
    while len(pos) < len(Y):
        pos.append(0)		#just for syntax...but we are not going to use these anyway; The assumption here is that the stacked bars are always the last
    hatches = hatches[:len(sys_legends)]
    width = 0.2
    figr = plt.figure(figsize=figsize)
    ps=[]
    for po, y,hatch, st in zip(pos, Y, hatches, stack):
        p = plt.bar(ind + po*width if st == -1 else ind + pos[st]*width, y, width,\
		 bottom=Y[st] if st != -1 else [0]*len(y), align='edge', hatch=hatch, edgecolor='black', lw=1.)
        ps.append(p)
    #Now, if there is some text, we should write it here
    if text is not None:
        #Note that: text should be also list of lists (the same dimensions of Y)
        for t,p in zip(text,ps):
            #t, p are lists
            for s, rect in zip(t,p):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, s, ha='center', va='bottom', weight='bold', color="red", fontsize=16)
    plt.ylabel(ylabel ,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.xticks(ind, xtick_labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(ps, sys_legends, fontsize=fontsize)
    plt.tight_layout()
    figr.savefig(f'{output_file}.pdf', bbox_inches = "tight")
    plt.gcf().clear()
