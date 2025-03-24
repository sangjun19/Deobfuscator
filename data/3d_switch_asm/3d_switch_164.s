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
	.ascii "1-2-1\0"
	.align	3
lC5:
	.ascii "1-2-2\0"
	.align	3
lC6:
	.ascii "1-2-3\0"
	.align	3
lC7:
	.ascii "1-2-4\0"
	.align	3
lC8:
	.ascii "1-3-1\0"
	.align	3
lC9:
	.ascii "1-3-2\0"
	.align	3
lC10:
	.ascii "1-3-3\0"
	.align	3
lC11:
	.ascii "1-3-4\0"
	.align	3
lC12:
	.ascii "1-4-1\0"
	.align	3
lC13:
	.ascii "1-4-2\0"
	.align	3
lC14:
	.ascii "1-4-3\0"
	.align	3
lC15:
	.ascii "1-4-4\0"
	.align	3
lC16:
	.ascii "1-5-1\0"
	.align	3
lC17:
	.ascii "1-5-2\0"
	.align	3
lC18:
	.ascii "1-5-3\0"
	.align	3
lC19:
	.ascii "1-5-4\0"
	.align	3
lC20:
	.ascii "1-6-1\0"
	.align	3
lC21:
	.ascii "1-6-2\0"
	.align	3
lC22:
	.ascii "1-6-3\0"
	.align	3
lC23:
	.ascii "1-6-4\0"
	.align	3
lC24:
	.ascii "1-7-1\0"
	.align	3
lC25:
	.ascii "1-7-2\0"
	.align	3
lC26:
	.ascii "1-7-3\0"
	.align	3
lC27:
	.ascii "1-7-4\0"
	.align	3
lC28:
	.ascii "2-1-1\0"
	.align	3
lC29:
	.ascii "2-1-2\0"
	.align	3
lC30:
	.ascii "2-1-3\0"
	.align	3
lC31:
	.ascii "2-1-4\0"
	.align	3
lC32:
	.ascii "2-2-1\0"
	.align	3
lC33:
	.ascii "2-2-2\0"
	.align	3
lC34:
	.ascii "2-2-3\0"
	.align	3
lC35:
	.ascii "2-2-4\0"
	.align	3
lC36:
	.ascii "2-3-1\0"
	.align	3
lC37:
	.ascii "2-3-2\0"
	.align	3
lC38:
	.ascii "2-3-3\0"
	.align	3
lC39:
	.ascii "2-3-4\0"
	.align	3
lC40:
	.ascii "2-4-1\0"
	.align	3
lC41:
	.ascii "2-4-2\0"
	.align	3
lC42:
	.ascii "2-4-3\0"
	.align	3
lC43:
	.ascii "2-4-4\0"
	.align	3
lC44:
	.ascii "2-5-1\0"
	.align	3
lC45:
	.ascii "2-5-2\0"
	.align	3
lC46:
	.ascii "2-5-3\0"
	.align	3
lC47:
	.ascii "2-5-4\0"
	.align	3
lC48:
	.ascii "2-6-1\0"
	.align	3
lC49:
	.ascii "2-6-2\0"
	.align	3
lC50:
	.ascii "2-6-3\0"
	.align	3
lC51:
	.ascii "2-6-4\0"
	.align	3
lC52:
	.ascii "2-7-1\0"
	.align	3
lC53:
	.ascii "2-7-2\0"
	.align	3
lC54:
	.ascii "2-7-3\0"
	.align	3
lC55:
	.ascii "2-7-4\0"
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
	mov	w0, 7
	str	w0, [x29, 24]
	mov	w0, 4
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
	cmp	w0, 7
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L12
	b	L92
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L93
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L93
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L93
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L14
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L14
L15:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L14
L13:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L14:
	b	L93
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L94
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L94
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L94
L21:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L19
L22:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L19
L20:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L19
L18:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L19:
	b	L94
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L95
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L95
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L27
	b	L95
L26:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L24
L27:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L24
L25:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L24
L23:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L24:
	b	L95
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L96
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L96
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L96
L31:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L29
L32:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L29
L30:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L29
L28:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L29:
	b	L96
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L97
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L97
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L37
	b	L97
L36:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L34
L37:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L34
L35:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L34
L33:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L34:
	b	L97
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L98
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L98
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L98
L41:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L39
L42:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L39
L40:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L39
L38:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L39:
	b	L98
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L99
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L99
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L47
	b	L99
L46:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L44
L47:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L44
L45:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L44
L43:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L44:
	b	L99
L93:
	nop
	b	L92
L94:
	nop
	b	L92
L95:
	nop
	b	L92
L96:
	nop
	b	L92
L97:
	nop
	b	L92
L98:
	nop
	b	L92
L99:
	nop
	b	L92
L3:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L48
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L100
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L50
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L100
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L100
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L100
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L100
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L55
	b	L100
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L101
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L101
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L101
L59:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L57
L60:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L57
L58:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L57
L56:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L57:
	b	L101
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L102
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L102
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L64
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L65
	b	L102
L64:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L62
L65:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L62
L63:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L62
L61:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L62:
	b	L102
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L66
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L103
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L103
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L70
	b	L103
L69:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L67
L70:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L67
L68:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L67
L66:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L67:
	b	L103
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L104
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L104
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L75
	b	L104
L74:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L72
L75:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	b	L72
L73:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L72
L71:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L72:
	b	L104
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L76
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L105
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L78
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L105
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L80
	b	L105
L79:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L77
L80:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L77
L78:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L77
L76:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L77:
	b	L105
L50:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L106
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L106
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L85
	b	L106
L84:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L82
L85:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L82
L83:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L82:
	b	L106
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L107
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L107
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L90
	b	L107
L89:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L87
L90:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	b	L87
L88:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L87
L86:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L87:
	b	L107
L101:
	nop
	b	L100
L102:
	nop
	b	L100
L103:
	nop
	b	L100
L104:
	nop
	b	L100
L105:
	nop
	b	L100
L106:
	nop
	b	L100
L107:
	nop
	b	L100
L92:
	nop
	b	L4
L100:
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
