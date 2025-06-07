#!/usr/bin/env python
# coding: utf-8

"""
Created on Saturday June 7 2025

@author: Pratik
"""
import torch
import torch.nn as nn


class ConvLSTM1DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        # x shape: (B, input_dim, L)
        combined = torch.cat([x, h_prev], dim=1)  # along channel dimension
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM1D_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvLSTM1D_Model, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.kernel_size = 3
        self.padding = 1

        self.lstm_cell = ConvLSTM1DCell(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.output_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: (B, C_in, L)
        B, _, L = x.size()
        device = x.device

        # Initialize hidden and cell states
        h = torch.zeros(B, self.hidden_dim, L, device=device)
        c = torch.zeros(B, self.hidden_dim, L, device=device)

        # One timestep (pseudo-single-step LSTM)
        h, c = self.lstm_cell(x, h, c)

        # Apply output layer
        out = self.output_conv(h).squeeze(1)  # shape: [B, L]
        return out


class ConvLSTM2DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        # x shape: (B, input_dim, H, W)
        combined = torch.cat([x, h_prev], dim=1)  # along channel dimension
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM2D_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvLSTM2D_Model, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.kernel_size = 3
        self.padding = 1

        self.lstm_cell = ConvLSTM2DCell(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        B, _, H, W = x.size()
        device = x.device

        # Initialize hidden and cell states
        h = torch.zeros(B, self.hidden_dim, H, W, device=device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=device)

        # One timestep (pseudo-single-step LSTM)
        h, c = self.lstm_cell(x, h, c)

        # Apply output layer
        out = self.output_conv(h)  # shape: [B, output_dim, H, W]
        return out
