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
	.ascii "2-1-1\0"
	.align	3
lC7:
	.ascii "2-2-1\0"
	.align	3
lC8:
	.ascii "2-3-1\0"
	.align	3
lC9:
	.ascii "2-4-1\0"
	.align	3
lC10:
	.ascii "2-5-1\0"
	.align	3
lC11:
	.ascii "2-6-1\0"
	.align	3
lC12:
	.ascii "3-1-1\0"
	.align	3
lC13:
	.ascii "3-2-1\0"
	.align	3
lC14:
	.ascii "3-3-1\0"
	.align	3
lC15:
	.ascii "3-4-1\0"
	.align	3
lC16:
	.ascii "3-5-1\0"
	.align	3
lC17:
	.ascii "3-6-1\0"
	.align	3
lC18:
	.ascii "4-1-1\0"
	.align	3
lC19:
	.ascii "4-2-1\0"
	.align	3
lC20:
	.ascii "4-3-1\0"
	.align	3
lC21:
	.ascii "4-4-1\0"
	.align	3
lC22:
	.ascii "4-5-1\0"
	.align	3
lC23:
	.ascii "4-6-1\0"
	.align	3
lC24:
	.ascii "5-1-1\0"
	.align	3
lC25:
	.ascii "5-2-1\0"
	.align	3
lC26:
	.ascii "5-3-1\0"
	.align	3
lC27:
	.ascii "5-4-1\0"
	.align	3
lC28:
	.ascii "5-5-1\0"
	.align	3
lC29:
	.ascii "5-6-1\0"
	.align	3
lC30:
	.ascii "6-1-1\0"
	.align	3
lC31:
	.ascii "6-2-1\0"
	.align	3
lC32:
	.ascii "6-3-1\0"
	.align	3
lC33:
	.ascii "6-4-1\0"
	.align	3
lC34:
	.ascii "6-5-1\0"
	.align	3
lC35:
	.ascii "6-6-1\0"
	.align	3
lC36:
	.ascii "7-1-1\0"
	.align	3
lC37:
	.ascii "7-2-1\0"
	.align	3
lC38:
	.ascii "7-3-1\0"
	.align	3
lC39:
	.ascii "7-4-1\0"
	.align	3
lC40:
	.ascii "7-5-1\0"
	.align	3
lC41:
	.ascii "7-6-1\0"
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
	mov	w0, 6
	str	w0, [x29, 24]
	mov	w0, 1
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
	cmp	w0, 6
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L16
	b	L102
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L103
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L103
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L104
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L104
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L105
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L105
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L106
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L106
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L107
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L107
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L108
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L108
L103:
	nop
	b	L102
L104:
	nop
	b	L102
L105:
	nop
	b	L102
L106:
	nop
	b	L102
L107:
	nop
	b	L102
L108:
	nop
	b	L102
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L23
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L109
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L109
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L109
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L109
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L29
	b	L109
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L110
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L110
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L111
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L111
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L112
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L112
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L113
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L113
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L114
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L114
L23:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L115
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L115
L110:
	nop
	b	L109
L111:
	nop
	b	L109
L112:
	nop
	b	L109
L113:
	nop
	b	L109
L114:
	nop
	b	L109
L115:
	nop
	b	L109
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L42
	b	L116
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L117
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L117
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L118
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L118
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L119
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L119
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L120
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L120
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L121
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L121
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L122
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L122
L117:
	nop
	b	L116
L118:
	nop
	b	L116
L119:
	nop
	b	L116
L120:
	nop
	b	L116
L121:
	nop
	b	L116
L122:
	nop
	b	L116
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L49
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L123
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L123
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L123
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L123
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L55
	b	L123
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L124
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L124
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L125
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L125
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L126
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L126
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L127
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L127
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L128
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L128
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L129
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L129
L124:
	nop
	b	L123
L125:
	nop
	b	L123
L126:
	nop
	b	L123
L127:
	nop
	b	L123
L128:
	nop
	b	L123
L129:
	nop
	b	L123
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L62
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L68
	b	L130
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L131
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
	b	L131
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L132
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
	b	L132
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L133
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
	b	L133
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L134
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
	b	L134
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L135
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	nop
	b	L135
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L136
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
	b	L136
L131:
	nop
	b	L130
L132:
	nop
	b	L130
L133:
	nop
	b	L130
L134:
	nop
	b	L130
L135:
	nop
	b	L130
L136:
	nop
	b	L130
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L137
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L137
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L78
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L137
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L79
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L137
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L80
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L81
	b	L137
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L138
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	nop
	b	L138
L81:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L139
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
	b	L139
L79:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L140
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
	b	L140
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L141
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
	b	L141
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L142
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
	b	L142
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L143
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
	b	L143
L138:
	nop
	b	L137
L139:
	nop
	b	L137
L140:
	nop
	b	L137
L141:
	nop
	b	L137
L142:
	nop
	b	L137
L143:
	nop
	b	L137
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L88
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L90
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L91
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L92
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L94
	b	L144
L93:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L145
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	nop
	b	L145
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L146
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
	b	L146
L92:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L147
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
	b	L147
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L148
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
	b	L148
L90:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L149
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	nop
	b	L149
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L150
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
	b	L150
L145:
	nop
	b	L144
L146:
	nop
	b	L144
L147:
	nop
	b	L144
L148:
	nop
	b	L144
L149:
	nop
	b	L144
L150:
	nop
	b	L144
L102:
	nop
	b	L3
L109:
	nop
	b	L3
L116:
	nop
	b	L3
L123:
	nop
	b	L3
L130:
	nop
	b	L3
L137:
	nop
	b	L3
L144:
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
