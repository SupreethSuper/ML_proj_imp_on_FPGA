 `include "GLOBAL_PARAMS.vh"
 `timescale 1ns/1ps
 
 module input_buffer #(
    parameter DATA_WIDTH = DATA_WIDTH,
    parameter ADDR_WIDTH = ADDR_WIDTH
 ) ();

 logic clk;
 logic rst_n;
 int counter;
 logic [ DATA_WIDTH - 1 : 0 ] rd_addr; //reader address

 //clock gen
 always begin : clock_gen
    clk <= 1'b1;
    #10;
    clk <= 1'b0;
 end

 
 always @(clk) begin 

    for(counter = 100; counter != 0; counter -- ) begin
        rst_n <= 1'b0;
        rst_n <= 1'b1;
    end


    
 end

 initial begin
    if(counter == 100) $finish;
 end

 initial begin
    $monitor("rd_addr = %h, rst_n = %h, clk = %h", rd_addr, rst_n, clk);
 end

 final begin
    $display("test called off from finish\n";)
 end




 endmodule