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

def plot_bars(Y, sys_legends, xtick_labels, hatches, xlabel, ylabel, output_file):
    #This function plots bars defined in Y (which is list of lists) for multiple systems (i.e., competitors)
    #sys_legends define the systems' names and xtick_labels define the labels on the x-axis
    assert len(Y) == len(sys_legends) and len(Y[0]) == len(xtick_labels)
    ind = np.arange(len(xtick_labels))
    #next, we calculate the positions of the bars (with align = 'left' always)
    start = -len(sys_legends)/2.0
    pos = list(np.arange(start,start*-1,1))
    hatches = hatches[:len(sys_legends)]
    width = 0.2
    figr = plt.figure(figsize=figsize)
    ps=[]
    for po, y,hatch in zip(pos, Y, hatches):
        p = plt.bar(ind + po*width, y, width, align='edge', hatch=hatch, edgecolor='black', lw=1.)
        ps.append(p)
    plt.ylabel(ylabel ,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.xticks(ind, xtick_labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(ps, sys_legends, fontsize=fontsize)
    plt.tight_layout()
    figr.savefig(f'{output_file}.pdf', bbox_inches = "tight")
    plt.gcf().clear()
