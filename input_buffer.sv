 `include "GLOBAL_PARAMS.vh"
 
 module input_buffer #(
    parameter DATA_WIDTH = DATA_WIDTH,
    parameter ADDR_WIDTH = ADDR_WIDTH
 )(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic rd_en,

    
    input logic [ ADDR_WIDTH - 1 : 0 ] wr_addr,
    input logic [ ADDR_WIDTH - 1 : 0 ] rd_addr,


    input  logic [ DATA_WIDTH - 1 : 0 ] wr_addr,

    output logic [ DATA_WIDTH - 1 : 0 ] rd_addr,

 );


    always_ff @( posedge clk or negedge rst_n ) begin : control_ff
        
        if(!rst_n) begin
            rd_addr = '0;
        end
        else begin
            
        end


    end
	
 
 
 endmodule