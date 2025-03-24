	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-2-1\0"
	.align	3
lC2:
	.ascii "1-3-1\0"
	.align	3
lC3:
	.ascii "1-4-1\0"
	.align	3
lC4:
	.ascii "1-5-1\0"
	.align	3
lC5:
	.ascii "1-6-1\0"
	.align	3
lC6:
	.ascii "1-7-1\0"
	.align	3
lC7:
	.ascii "1-8-1\0"
	.align	3
lC8:
	.ascii "1-9-1\0"
	.align	3
lC9:
	.ascii "2-1-1\0"
	.align	3
lC10:
	.ascii "2-2-1\0"
	.align	3
lC11:
	.ascii "2-3-1\0"
	.align	3
lC12:
	.ascii "2-4-1\0"
	.align	3
lC13:
	.ascii "2-5-1\0"
	.align	3
lC14:
	.ascii "2-6-1\0"
	.align	3
lC15:
	.ascii "2-7-1\0"
	.align	3
lC16:
	.ascii "2-8-1\0"
	.align	3
lC17:
	.ascii "2-9-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 2
	str	w0, [x29, 28]
	mov	w0, 9
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L3
	b	L4
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L44
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L44
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L44
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L44
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L44
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L44
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L44
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L44
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L45
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L45
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L46
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L46
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L47
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L47
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L48
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L48
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L49
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L49
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L50
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L50
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L51
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L51
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L52
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L52
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L53
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L53
L45:
	nop
	b	L44
L46:
	nop
	b	L44
L47:
	nop
	b	L44
L48:
	nop
	b	L44
L49:
	nop
	b	L44
L50:
	nop
	b	L44
L51:
	nop
	b	L44
L52:
	nop
	b	L44
L53:
	nop
	b	L44
L3:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L24
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L54
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L54
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L54
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L54
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L54
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L54
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L54
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L33
	b	L54
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L55
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L55
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L56
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L56
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L57
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L57
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L58
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L58
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L59
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L59
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L60
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L60
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L61
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L61
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L62
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L62
L24:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L63
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L63
L55:
	nop
	b	L54
L56:
	nop
	b	L54
L57:
	nop
	b	L54
L58:
	nop
	b	L54
L59:
	nop
	b	L54
L60:
	nop
	b	L54
L61:
	nop
	b	L54
L62:
	nop
	b	L54
L63:
	nop
	b	L54
L44:
	nop
	b	L4
L54:
	nop
L4:
	mov	w0, 0
	ldp	x29, x30, [sp], 32
LCFI2:
	ret
LFE1:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.uleb128 0x1
	.byte	0x10
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB1-.
	.set L$set$2,LFE1-LFB1
	.quad L$set$2
	.uleb128 0
	.byte	0x4
	.set L$set$3,LCFI0-LFB1
	.long L$set$3
	.byte	0xe
	.uleb128 0x20
	.byte	0x9d
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x3
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE1:
	.ident	"GCC: (Homebrew GCC 14.2.0_1) 14.2.0"
	.subsections_via_symbols
