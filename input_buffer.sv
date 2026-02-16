 `include "GLOBAL_PARAMS.vh"
 
 module input_buffer #(
    parameter BUFFER_DATA_WIDTH = DATA_WIDTH,
    parameter BUFFER_ADDR_WIDTH = ADDR_WIDTH
 )(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic rd_en,

    
    input logic [ BUFFER_ADDR_WIDTH - 1 : 0 ] wr_addr,
    input logic [ BUFFER_ADDR_WIDTH - 1 : 0 ] rd_addr,


    input  logic [ BUFFER_DATA_WIDTH - 1 : 0 ] wr_data,

    output logic [ BUFFER_DATA_WIDTH - 1 : 0 ] rd_data

 );


    always_ff @( posedge clk or negedge rst_n ) begin : control_ff
        

        //async reset, triiger at rst low
        if(!rst_n) begin
            rd_data = '0;
        end
        else begin
            //logic to be loaded
        end


    end
	
 
 
 endmodule