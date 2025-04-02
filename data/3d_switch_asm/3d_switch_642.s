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
	.ascii "1-2-1\0"
	.align	3
lC3:
	.ascii "1-2-2\0"
	.align	3
lC4:
	.ascii "1-3-1\0"
	.align	3
lC5:
	.ascii "1-3-2\0"
	.align	3
lC6:
	.ascii "1-4-1\0"
	.align	3
lC7:
	.ascii "1-4-2\0"
	.align	3
lC8:
	.ascii "1-5-1\0"
	.align	3
lC9:
	.ascii "1-5-2\0"
	.align	3
lC10:
	.ascii "2-1-1\0"
	.align	3
lC11:
	.ascii "2-1-2\0"
	.align	3
lC12:
	.ascii "2-2-1\0"
	.align	3
lC13:
	.ascii "2-2-2\0"
	.align	3
lC14:
	.ascii "2-3-1\0"
	.align	3
lC15:
	.ascii "2-3-2\0"
	.align	3
lC16:
	.ascii "2-4-1\0"
	.align	3
lC17:
	.ascii "2-4-2\0"
	.align	3
lC18:
	.ascii "2-5-1\0"
	.align	3
lC19:
	.ascii "2-5-2\0"
	.align	3
lC20:
	.ascii "3-1-1\0"
	.align	3
lC21:
	.ascii "3-1-2\0"
	.align	3
lC22:
	.ascii "3-2-1\0"
	.align	3
lC23:
	.ascii "3-2-2\0"
	.align	3
lC24:
	.ascii "3-3-1\0"
	.align	3
lC25:
	.ascii "3-3-2\0"
	.align	3
lC26:
	.ascii "3-4-1\0"
	.align	3
lC27:
	.ascii "3-4-2\0"
	.align	3
lC28:
	.ascii "3-5-1\0"
	.align	3
lC29:
	.ascii "3-5-2\0"
	.align	3
lC30:
	.ascii "4-1-1\0"
	.align	3
lC31:
	.ascii "4-1-2\0"
	.align	3
lC32:
	.ascii "4-2-1\0"
	.align	3
lC33:
	.ascii "4-2-2\0"
	.align	3
lC34:
	.ascii "4-3-1\0"
	.align	3
lC35:
	.ascii "4-3-2\0"
	.align	3
lC36:
	.ascii "4-4-1\0"
	.align	3
lC37:
	.ascii "4-4-2\0"
	.align	3
lC38:
	.ascii "4-5-1\0"
	.align	3
lC39:
	.ascii "4-5-2\0"
	.align	3
lC40:
	.ascii "5-1-1\0"
	.align	3
lC41:
	.ascii "5-1-2\0"
	.align	3
lC42:
	.ascii "5-2-1\0"
	.align	3
lC43:
	.ascii "5-2-2\0"
	.align	3
lC44:
	.ascii "5-3-1\0"
	.align	3
lC45:
	.ascii "5-3-2\0"
	.align	3
lC46:
	.ascii "5-4-1\0"
	.align	3
lC47:
	.ascii "5-4-2\0"
	.align	3
lC48:
	.ascii "5-5-1\0"
	.align	3
lC49:
	.ascii "5-5-2\0"
	.align	3
lC50:
	.ascii "6-1-1\0"
	.align	3
lC51:
	.ascii "6-1-2\0"
	.align	3
lC52:
	.ascii "6-2-1\0"
	.align	3
lC53:
	.ascii "6-2-2\0"
	.align	3
lC54:
	.ascii "6-3-1\0"
	.align	3
lC55:
	.ascii "6-3-2\0"
	.align	3
lC56:
	.ascii "6-4-1\0"
	.align	3
lC57:
	.ascii "6-4-2\0"
	.align	3
lC58:
	.ascii "6-5-1\0"
	.align	3
lC59:
	.ascii "6-5-2\0"
	.align	3
lC60:
	.ascii "7-1-1\0"
	.align	3
lC61:
	.ascii "7-1-2\0"
	.align	3
lC62:
	.ascii "7-2-1\0"
	.align	3
lC63:
	.ascii "7-2-2\0"
	.align	3
lC64:
	.ascii "7-3-1\0"
	.align	3
lC65:
	.ascii "7-3-2\0"
	.align	3
lC66:
	.ascii "7-4-1\0"
	.align	3
lC67:
	.ascii "7-4-2\0"
	.align	3
lC68:
	.ascii "7-5-1\0"
	.align	3
lC69:
	.ascii "7-5-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 7
	str	w0, [x29, 28]
	mov	w0, 5
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L9
	b	L3
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L15
	b	L158
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L11
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L11
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L11
L19:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L21
L20:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L21:
	b	L11
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L11
L22:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L24
L23:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L24:
	b	L11
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L11
L25:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L27
L26:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L27:
	b	L11
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L29
	b	L159
L28:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L30:
L159:
	nop
L11:
	b	L158
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L36
	b	L160
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L38
	b	L32
L37:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L39
L38:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L39:
	b	L32
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L41
	b	L32
L40:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L42
L41:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L42:
	b	L32
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L44
	b	L32
L43:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L45
L44:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L45:
	b	L32
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L47
	b	L32
L46:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L32
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L161
L49:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L51
L50:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L51:
L161:
	nop
L32:
	b	L160
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L55
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L57
	b	L162
L56:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L59
	b	L53
L58:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L60
L59:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L60:
	b	L53
L57:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L62
	b	L53
L61:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L63
L62:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L63:
	b	L53
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L64
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L65
	b	L53
L64:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L66
L65:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L66:
	b	L53
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L67
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L68
	b	L53
L67:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L69
L68:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L69:
	b	L53
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L71
	b	L163
L70:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L72
L71:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L72:
L163:
	nop
L53:
	b	L162
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L73
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L78
	b	L164
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L80
	b	L74
L79:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L81
L80:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L81:
	b	L74
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L83
	b	L74
L82:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L84
L83:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L84:
	b	L74
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L86
	b	L74
L85:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L87
L86:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L87:
	b	L74
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L89
	b	L74
L88:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L90
L89:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L90:
	b	L74
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L91
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L92
	b	L165
L91:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L93
L92:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L93:
L165:
	nop
L74:
	b	L164
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L94
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L96
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L97
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L98
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L99
	b	L166
L98:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L100
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L101
	b	L95
L100:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L102
L101:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L102:
	b	L95
L99:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L104
	b	L95
L103:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L105
L104:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L105:
	b	L95
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L106
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L107
	b	L95
L106:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L108
L107:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L108:
	b	L95
L96:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L110
	b	L95
L109:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L111
L110:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L111:
	b	L95
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L112
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L113
	b	L167
L112:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L114
L113:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L114:
L167:
	nop
L95:
	b	L166
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L115
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L117
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L118
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L119
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L120
	b	L168
L119:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L121
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L122
	b	L116
L121:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L123
L122:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L123:
	b	L116
L120:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L124
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L125
	b	L116
L124:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L126
L125:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L126:
	b	L116
L118:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L127
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L128
	b	L116
L127:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L129
L128:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L129:
	b	L116
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L130
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L131
	b	L116
L130:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L132
L131:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L132:
	b	L116
L115:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L133
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L134
	b	L169
L133:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L135
L134:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L135:
L169:
	nop
L116:
	b	L168
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L136
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L170
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L138
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L170
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L139
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L170
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L140
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L141
	b	L170
L140:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L142
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L143
	b	L137
L142:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L144
L143:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	nop
L144:
	b	L137
L141:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L145
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L146
	b	L137
L145:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L147
L146:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L147:
	b	L137
L139:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L148
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L149
	b	L137
L148:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L150
L149:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	nop
L150:
	b	L137
L138:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L151
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L152
	b	L137
L151:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L153
L152:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
L153:
	b	L137
L136:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L154
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L155
	b	L171
L154:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L156
L155:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	nop
L156:
L171:
	nop
L137:
	b	L170
L158:
	nop
	b	L3
L160:
	nop
	b	L3
L162:
	nop
	b	L3
L164:
	nop
	b	L3
L166:
	nop
	b	L3
L168:
	nop
	b	L3
L170:
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
