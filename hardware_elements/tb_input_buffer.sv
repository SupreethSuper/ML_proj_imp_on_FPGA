// tb_input_buffer.sv
// Simple directed + light random testbench for YOUR exact input_buffer (no DUT changes).
// Notes:
// - Constrains addresses to DUT's *actual* mem depth: DEPTH = (2*BUFFER_ADDR_WIDTH)
//   because DUT declares: mem[(2*ADDR_WIDTH)-1:0]
// - Avoids fatal asserts by default (your DUT has some known oddities on reset/collision).
// - Drives inputs on negedge, samples on posedge (clean setup/hold for synchronous logic).

`timescale 1ns/1ps
`include "GLOBAL_PARAMS.vh"

module tb_input_buffer;

  // Use the same widths as DUT defaults.
  localparam int DW = DATA_WIDTH;
  localparam int AW = ADDR_WIDTH;

  // IMPORTANT: Match DUT's current mem depth expression (2*ADDR_WIDTH)
  localparam int DEPTH = (2 * AW);

  logic clk;
  logic rst_n;

  logic wr_en;
  logic rd_en;

  logic [AW-1:0] wr_addr;
  logic [AW-1:0] rd_addr;

  logic signed [DW-1:0] wr_data;
  logic signed [DW-1:0] rd_data;

  // DUT instance (no parameter override needed)
  input_buffer dut (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_en   (wr_en),
    .rd_en   (rd_en),
    .wr_addr (wr_addr),
    .rd_addr (rd_addr),
    .wr_data (wr_data),
    .rd_data (rd_data)
  );

  // Clock: 100MHz equivalent (10ns period)
  initial clk = 1'b0;
  always  #5 clk = ~clk;

  // Simple shadow model of memory for non-collision checks
  logic signed [DW-1:0] model_mem [0:DEPTH-1];

  // Helper: constrain random addresses to DUT mem range [0..DEPTH-1]
  function automatic [AW-1:0] rand_addr_in_range();
    int unsigned a;
    begin
      a = $urandom_range(0, DEPTH-1);
      rand_addr_in_range = a[AW-1:0];
    end
  endfunction

  // Drive defaults
  task automatic drive_idle();
    begin
      wr_en   = 1'b0;
      rd_en   = 1'b0;
      wr_addr = '0;
      rd_addr = '0;
      wr_data = '0;
    end
  endtask

  // Apply async reset (active low)
  task automatic apply_async_reset();
    begin
      rst_n = 1'b1;
      @(negedge clk);
      rst_n = 1'b0;
      // Hold reset low for a bit (async)
      #2;
      // release near negedge for clean sampling
      @(negedge clk);
      rst_n = 1'b1;
    end
  endtask

  // Write one word (setup on negedge, commit on next posedge)
  task automatic do_write(input [AW-1:0] a, input logic signed [DW-1:0] d);
    begin
      @(negedge clk);
      wr_en   = 1'b1;
      wr_addr = a;
      wr_data = d;
      rd_en   = 1'b0;
      rd_addr = '0;

      @(posedge clk); // write occurs here (in DUT)
      // Update model (ONLY if address is in range)
      if (a < DEPTH) model_mem[a] = d;

      @(negedge clk);
      wr_en = 1'b0;
    end
  endtask

  // Read one word (setup on negedge, capture on next posedge)
  task automatic do_read(input [AW-1:0] a, output logic signed [DW-1:0] q);
    begin
      @(negedge clk);
      rd_en   = 1'b1;
      rd_addr = a;
      wr_en   = 1'b0;
      wr_addr = '0;
      wr_data = '0;

      @(posedge clk);
      // sample after the posedge update
      #1;
      q = rd_data;

      @(negedge clk);
      rd_en = 1'b0;
    end
  endtask

  // Simultaneous read+write (same cycle) to test collision behavior
  task automatic do_simul_rw(
      input [AW-1:0] wa, input logic signed [DW-1:0] wd,
      input [AW-1:0] ra, output logic signed [DW-1:0] rq
  );
    begin
      @(negedge clk);
      wr_en   = 1'b1;  wr_addr = wa; wr_data = wd;
      rd_en   = 1'b1;  rd_addr = ra;

      @(posedge clk);
      #1;
      rq = rd_data;

      // Update model only if wa in range; collision policy is DUT-defined (we won't enforce here)
      if (wa < DEPTH) model_mem[wa] = wd;

      @(negedge clk);
      wr_en = 1'b0;
      rd_en = 1'b0;
    end
  endtask

  // Pretty print helper
  task automatic print_status(string tag);
    $display("[%0t] %s | rst_n=%0b wr_en=%0b rd_en=%0b wa=%0d ra=%0d wd=%0d rd=%0d",
             $time, tag, rst_n, wr_en, rd_en, wr_addr, rd_addr, wr_data, rd_data);
  endtask

  int i;
  int mismatches;

  initial begin
    mismatches = 0;

    // Initialize model memory to known values
    for (i = 0; i < DEPTH; i++) model_mem[i] = '0;

    drive_idle();
    rst_n = 1'b1;

    $display("============================================================");
    $display("TB START: input_buffer");
    $display("DW=%0d AW=%0d DEPTH(according to DUT)=%0d", DW, AW, DEPTH);
    $display("NOTE: TB constrains addresses to [0..DEPTH-1] because DUT mem depth is (2*AW).");
    $display("============================================================");

    // --------------------------
    // TEST 0: Reset sanity
    // --------------------------
    apply_async_reset();
    print_status("After reset release");

    // Expectation (ideal): rd_data=0 after reset.
    // But your DUT assigns rd_data=mem[rd_addr] during reset, so this may become X.
    // We only report.
    if (^rd_data === 1'bX) begin
      $display("[WARN] rd_data is X after reset (expected from current reset logic).");
    end else if (rd_data !== '0) begin
      $display("[WARN] rd_data != 0 after reset: rd_data=%0d", rd_data);
    end

    // --------------------------
    // TEST 1: Simple write/read
    // --------------------------
    begin
      logic signed [DW-1:0] q;
      logic [AW-1:0] a;
      a = 0;
      do_write(a, 16'sd5);
      do_read(a, q);
      $display("[T1] Readback addr=%0d => %0d (model=%0d)", a, q, model_mem[a]);
      if (q !== model_mem[a]) begin
        $display("[MISMATCH T1] addr=%0d got=%0d expected=%0d", a, q, model_mem[a]);
        mismatches++;
      end
    end

    // --------------------------
    // TEST 2: Negative value write/read
    // --------------------------
    begin
      logic signed [DW-1:0] q;
      logic [AW-1:0] a;
      a = (DEPTH > 3) ? 3 : 0;
      do_write(a, -16'sd7);
      do_read(a, q);
      $display("[T2] Readback addr=%0d => %0d (model=%0d)", a, q, model_mem[a]);
      if (q !== model_mem[a]) begin
        $display("[MISMATCH T2] addr=%0d got=%0d expected=%0d", a, q, model_mem[a]);
        mismatches++;
      end
    end

    // --------------------------
    // TEST 3: Back-to-back reads
    // --------------------------
    begin
      logic signed [DW-1:0] q1, q2;
      logic [AW-1:0] a1, a2;
      a1 = 0;
      a2 = (DEPTH > 3) ? 3 : 0;
      do_read(a1, q1);
      do_read(a2, q2);
      $display("[T3] q1(addr=%0d)=%0d model=%0d | q2(addr=%0d)=%0d model=%0d",
               a1, q1, model_mem[a1], a2, q2, model_mem[a2]);
      if (q1 !== model_mem[a1]) begin mismatches++; $display("[MISMATCH T3-1]"); end
      if (q2 !== model_mem[a2]) begin mismatches++; $display("[MISMATCH T3-2]"); end
    end

    // --------------------------
    // TEST 4: Simultaneous R/W different addresses
    // --------------------------
    begin
      logic signed [DW-1:0] q;
      logic [AW-1:0] wa, ra;
      wa = (DEPTH > 5) ? 5 : 0;
      ra = 0;
      do_simul_rw(wa, 16'sd99, ra, q);
      $display("[T4] simul RW: wrote addr=%0d=99, read addr=%0d -> %0d (model=%0d)",
               wa, ra, q, model_mem[ra]);
      // In a clean dual-port, read should match model at ra (unless your collision logic interferes).
      if (q !== model_mem[ra]) begin
        $display("[MISMATCH T4] read got=%0d expected=%0d", q, model_mem[ra]);
        mismatches++;
      end
    end

    // --------------------------
    // TEST 5: Simultaneous R/W same address (collision observation)
    // --------------------------
    begin
      logic signed [DW-1:0] q;
      logic [AW-1:0] a;
      a = 0;
      // ensure known prior value
      do_write(a, 16'sd11);
      // now collide
      do_simul_rw(a, 16'sd22, a, q);
      $display("[T5] COLLISION same addr=%0d: prior=11, wrote=22, read_observed=%0d (post_model=%0d)",
               a, q, model_mem[a]);
      $display("     This test is informational: collision mode is DUT-defined.");
    end

    // --------------------------
    // TEST 6: Light constrained random (200 cycles)
    // - Only checks reads on NON-collision cycles
    // --------------------------
    $display("---- Random test: 200 cycles (non-collision read checks) ----");
    for (i = 0; i < 200; i++) begin
      logic do_wr, do_rd;
      logic [AW-1:0] wa, ra;
      logic signed [DW-1:0] wd;
      logic signed [DW-1:0] q;

      do_wr = $urandom_range(0, 1);
      do_rd = $urandom_range(0, 1);

      wa = rand_addr_in_range();
      ra = rand_addr_in_range();
      wd = $signed($urandom());

      // Drive both enables and addresses on negedge
      @(negedge clk);
      wr_en   = do_wr;
      rd_en   = do_rd;
      wr_addr = wa;
      rd_addr = ra;
      wr_data = wd;

      @(posedge clk);
      #1;
      q = rd_data;

      // Update model write
      if (do_wr && (wa < DEPTH)) model_mem[wa] = wd;

      // Only check read if:
      // - do_rd is asserted
      // - NOT (do_wr && do_rd && same address) to avoid collision ambiguity
      if (do_rd && !(do_wr && do_rd && (wa == ra))) begin
        if (q !== model_mem[ra]) begin
          $display("[RAND MISMATCH] cyc=%0d ra=%0d got=%0d expected=%0d | do_wr=%0b wa=%0d wd=%0d",
                   i, ra, q, model_mem[ra], do_wr, wa, wd);
          mismatches++;
        end
      end
    end
    @(negedge clk);
    drive_idle();

    $display("============================================================");
    $display("TB DONE. mismatches=%0d", mismatches);
    $display("If mismatches>0, inspect reset logic, mem depth, <= vs =, collision handling.");
    $display("============================================================");

    $finish;
  end

endmodule
