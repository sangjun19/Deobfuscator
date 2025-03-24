	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-1-2\0"
	.align	3
lC2:
	.ascii "1-1-3\0"
	.align	3
lC3:
	.ascii "1-1-4\0"
	.align	3
lC4:
	.ascii "1-1-5\0"
	.align	3
lC5:
	.ascii "1-1-6\0"
	.align	3
lC6:
	.ascii "1-1-7\0"
	.align	3
lC7:
	.ascii "1-2-1\0"
	.align	3
lC8:
	.ascii "1-2-2\0"
	.align	3
lC9:
	.ascii "1-2-3\0"
	.align	3
lC10:
	.ascii "1-2-4\0"
	.align	3
lC11:
	.ascii "1-2-5\0"
	.align	3
lC12:
	.ascii "1-2-6\0"
	.align	3
lC13:
	.ascii "1-2-7\0"
	.align	3
lC14:
	.ascii "1-3-1\0"
	.align	3
lC15:
	.ascii "1-3-2\0"
	.align	3
lC16:
	.ascii "1-3-3\0"
	.align	3
lC17:
	.ascii "1-3-4\0"
	.align	3
lC18:
	.ascii "1-3-5\0"
	.align	3
lC19:
	.ascii "1-3-6\0"
	.align	3
lC20:
	.ascii "1-3-7\0"
	.align	3
lC21:
	.ascii "2-1-1\0"
	.align	3
lC22:
	.ascii "2-1-2\0"
	.align	3
lC23:
	.ascii "2-1-3\0"
	.align	3
lC24:
	.ascii "2-1-4\0"
	.align	3
lC25:
	.ascii "2-1-5\0"
	.align	3
lC26:
	.ascii "2-1-6\0"
	.align	3
lC27:
	.ascii "2-1-7\0"
	.align	3
lC28:
	.ascii "2-2-1\0"
	.align	3
lC29:
	.ascii "2-2-2\0"
	.align	3
lC30:
	.ascii "2-2-3\0"
	.align	3
lC31:
	.ascii "2-2-4\0"
	.align	3
lC32:
	.ascii "2-2-5\0"
	.align	3
lC33:
	.ascii "2-2-6\0"
	.align	3
lC34:
	.ascii "2-2-7\0"
	.align	3
lC35:
	.ascii "2-3-1\0"
	.align	3
lC36:
	.ascii "2-3-2\0"
	.align	3
lC37:
	.ascii "2-3-3\0"
	.align	3
lC38:
	.ascii "2-3-4\0"
	.align	3
lC39:
	.ascii "2-3-5\0"
	.align	3
lC40:
	.ascii "2-3-6\0"
	.align	3
lC41:
	.ascii "2-3-7\0"
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
	mov	w0, 3
	str	w0, [x29, 24]
	mov	w0, 7
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
	cmp	w0, 3
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L62
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L8
	b	L62
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L9
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L63
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L11
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L63
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L63
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L63
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L63
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L63
L15:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L10
L16:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L10
L14:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L10
L13:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L10
L12:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L10
L11:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L10
L9:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
L10:
	b	L63
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L64
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L64
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L64
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L64
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L64
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L64
L23:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L18
L24:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L18
L22:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L18
L21:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L18
L20:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	b	L18
L19:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L64
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L65
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L65
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L65
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L65
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L65
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L65
L31:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L26
L32:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L26
L30:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L26
L29:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L26
L28:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L26
L27:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L26
L25:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L26:
	b	L65
L63:
	nop
	b	L62
L64:
	nop
	b	L62
L65:
	nop
	b	L62
L3:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L66
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L36
	b	L66
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L67
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L67
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L67
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L67
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L67
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L44
	b	L67
L43:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L38
L44:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L38
L42:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	b	L38
L41:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L38
L40:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L38
L39:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L38
L37:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L38:
	b	L67
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L68
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L68
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L68
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L68
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L68
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L52
	b	L68
L51:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L46
L52:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L46
L50:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L46
L49:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L46
L48:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L46
L47:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L46
L45:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
L46:
	b	L68
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L69
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L69
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L69
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L69
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L69
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L69
L59:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	b	L54
L60:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L54
L58:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L54
L57:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L54
L56:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	b	L54
L55:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L54
L53:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L54:
	b	L69
L67:
	nop
	b	L66
L68:
	nop
	b	L66
L69:
	nop
	b	L66
L62:
	nop
	b	L4
L66:
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
