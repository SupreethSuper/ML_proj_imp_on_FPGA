`include "GLOBAL_PARAMS.vh"
`timescale 1ns/1ps

module tb_input_buffer;

  localparam int BUFFER_DATA_WIDTH = DATA_WIDTH;
  localparam int BUFFER_ADDR_WIDTH = ADDR_WIDTH;

  logic clk;
  logic rst_n;

  // other signals...
  logic wr_en, rd_en;
  logic [BUFFER_ADDR_WIDTH-1:0] wr_addr, rd_addr;
  logic [BUFFER_DATA_WIDTH-1:0] wr_data, rd_data;

  input_buffer #(
    .BUFFER_DATA_WIDTH(BUFFER_DATA_WIDTH),
    .BUFFER_ADDR_WIDTH(BUFFER_ADDR_WIDTH)
  ) uut (
    .rst_n(rst_n),
    .clk(clk),
    .wr_en(wr_en),
    .rd_en(rd_en),
    .wr_addr(wr_addr),
    .rd_addr(rd_addr),
    .wr_data(wr_data),
    .rd_data(rd_data)
  );

  // clock
  initial clk = 1'b0;
  always #10 clk = ~clk;

  // reset + basic init
  initial begin
    rst_n  = 1'b0;
    wr_en  = 1'b0;
    rd_en  = 1'b0;
    wr_addr = '0;
    rd_addr = '0;
    wr_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
  end

  // finish after some cycles
  initial begin
    repeat (200) @(posedge clk);
    $finish;
  end

  initial $monitor("t=%0t clk=%b rst_n=%b rd_addr=%h", $time, clk, rst_n, rd_addr);

endmodule
