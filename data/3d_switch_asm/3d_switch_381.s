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
	.align	3
lC18:
	.ascii "3-1-1\0"
	.align	3
lC19:
	.ascii "3-2-1\0"
	.align	3
lC20:
	.ascii "3-3-1\0"
	.align	3
lC21:
	.ascii "3-4-1\0"
	.align	3
lC22:
	.ascii "3-5-1\0"
	.align	3
lC23:
	.ascii "3-6-1\0"
	.align	3
lC24:
	.ascii "3-7-1\0"
	.align	3
lC25:
	.ascii "3-8-1\0"
	.align	3
lC26:
	.ascii "3-9-1\0"
	.align	3
lC27:
	.ascii "4-1-1\0"
	.align	3
lC28:
	.ascii "4-2-1\0"
	.align	3
lC29:
	.ascii "4-3-1\0"
	.align	3
lC30:
	.ascii "4-4-1\0"
	.align	3
lC31:
	.ascii "4-5-1\0"
	.align	3
lC32:
	.ascii "4-6-1\0"
	.align	3
lC33:
	.ascii "4-7-1\0"
	.align	3
lC34:
	.ascii "4-8-1\0"
	.align	3
lC35:
	.ascii "4-9-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 4
	str	w0, [x29, 28]
	mov	w0, 9
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L6
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L16
	b	L84
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L85
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L85
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L86
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L86
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L87
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L87
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L88
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L88
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L89
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L89
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L90
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L90
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L91
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L91
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L92
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L92
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L93
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L93
L85:
	nop
	b	L84
L86:
	nop
	b	L84
L87:
	nop
	b	L84
L88:
	nop
	b	L84
L89:
	nop
	b	L84
L90:
	nop
	b	L84
L91:
	nop
	b	L84
L92:
	nop
	b	L84
L93:
	nop
	b	L84
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L35
	b	L94
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L95
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L95
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L96
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L96
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L97
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L97
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L98
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L98
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L99
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L99
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L100
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L100
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L101
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L101
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L102
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L102
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L103
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L103
L95:
	nop
	b	L94
L96:
	nop
	b	L94
L97:
	nop
	b	L94
L98:
	nop
	b	L94
L99:
	nop
	b	L94
L100:
	nop
	b	L94
L101:
	nop
	b	L94
L102:
	nop
	b	L94
L103:
	nop
	b	L94
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L104
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L47
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L104
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L48
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L104
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L49
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L104
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L50
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L104
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L104
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L104
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L54
	b	L104
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L105
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L105
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L106
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L106
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L107
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L107
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L108
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L108
L50:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L109
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L109
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L110
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L110
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L111
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
	b	L111
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L112
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
	b	L112
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L113
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
	b	L113
L105:
	nop
	b	L104
L106:
	nop
	b	L104
L107:
	nop
	b	L104
L108:
	nop
	b	L104
L109:
	nop
	b	L104
L110:
	nop
	b	L104
L111:
	nop
	b	L104
L112:
	nop
	b	L104
L113:
	nop
	b	L104
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L68
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L69
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L71
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L72
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L73
	b	L114
L72:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L115
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
	b	L115
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L116
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	nop
	b	L116
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L117
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
	b	L117
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L118
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	nop
	b	L118
L69:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L119
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
	b	L119
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L120
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
	b	L120
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L121
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
	b	L121
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L122
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
	b	L122
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L123
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
	b	L123
L115:
	nop
	b	L114
L116:
	nop
	b	L114
L117:
	nop
	b	L114
L118:
	nop
	b	L114
L119:
	nop
	b	L114
L120:
	nop
	b	L114
L121:
	nop
	b	L114
L122:
	nop
	b	L114
L123:
	nop
	b	L114
L84:
	nop
	b	L3
L94:
	nop
	b	L3
L104:
	nop
	b	L3
L114:
	nop
L3:
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
