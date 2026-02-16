// 
//  * @module input_buffer
//  * @brief Dual-port RAM buffer for input data storage and retrieval
//  * 
//  * This module implements a synchronous input buffer with independent read and write ports.
//  * It supports asynchronous reset and provides registered output data.
//  * 
//  * @param BUFFER_DATA_WIDTH - Width of data words stored in buffer (default: DATA_WIDTH)
//  * @param BUFFER_ADDR_WIDTH - Width of address bus (default: ADDR_WIDTH)
//  * 
//  * @port clk - System clock (posedge triggered)
//  * @port rst_n - Asynchronous active-low reset
//  * @port wr_en - Write enable signal
//  * @port rd_en - Read enable signal
//  * @port wr_addr - Write address
//  * @port rd_addr - Read address
//  * @port wr_data - Data to write into buffer
//  * @port rd_data - Data read from buffer (registered output)
//  * 
//  * @behavior
//  *   - On rst_n LOW: rd_data is asynchronously cleared to zero
//  *   - On clk POSEDGE: Output data is updated based on read operations
//  *   - Supports simultaneous independent read and write operations
//  
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