	.file	"mblaszczykowski_algorithms-data-structures-c_5_flatten.c"
	.text
	.globl	_TIG_IZ_i38W_envp
	.bss
	.align 8
	.type	_TIG_IZ_i38W_envp, @object
	.size	_TIG_IZ_i38W_envp, 8
_TIG_IZ_i38W_envp:
	.zero	8
	.globl	_TIG_IZ_i38W_argc
	.align 4
	.type	_TIG_IZ_i38W_argc, @object
	.size	_TIG_IZ_i38W_argc, 4
_TIG_IZ_i38W_argc:
	.zero	4
	.globl	_TIG_IZ_i38W_argv
	.align 8
	.type	_TIG_IZ_i38W_argv, @object
	.size	_TIG_IZ_i38W_argv, 8
_TIG_IZ_i38W_argv:
	.zero	8
	.globl	top
	.align 8
	.type	top, @object
	.size	top, 8
top:
	.zero	8
	.text
	.globl	empty
	.type	empty, @function
empty:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L9:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L10
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L10
	movq	top(%rip), %rax
	testq	%rax, %rax
	jne	.L5
	movq	$2, -8(%rbp)
	jmp	.L7
.L5:
	movq	$0, -8(%rbp)
	jmp	.L7
.L4:
	movl	$0, %eax
	jmp	.L8
.L2:
	movl	$1, %eax
	jmp	.L8
.L10:
	nop
.L7:
	jmp	.L9
.L8:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	empty, .-empty
	.section	.rodata
.LC0:
	.string	"Stos jest pusty"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L20:
	cmpq	$3, -8(%rbp)
	je	.L21
	cmpq	$3, -8(%rbp)
	ja	.L22
	cmpq	$2, -8(%rbp)
	je	.L14
	cmpq	$2, -8(%rbp)
	ja	.L22
	cmpq	$0, -8(%rbp)
	je	.L15
	cmpq	$1, -8(%rbp)
	jne	.L22
	movq	top(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, top(%rip)
	movq	$3, -8(%rbp)
	jmp	.L16
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L16
.L14:
	movq	top(%rip), %rax
	testq	%rax, %rax
	jne	.L18
	movq	$0, -8(%rbp)
	jmp	.L16
.L18:
	movq	$1, -8(%rbp)
	jmp	.L16
.L22:
	nop
.L16:
	jmp	.L20
.L21:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	pop, .-pop
	.section	.rodata
	.align 8
.LC1:
	.string	"Wpisz rownanie bez spacji np. '12+a*(b*c+d/e)' : "
.LC2:
	.string	"%s"
	.align 8
.LC3:
	.string	"Rownanie po zastosowaniu ONP: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, top(%rip)
	nop
.L24:
	movq	$0, _TIG_IZ_i38W_envp(%rip)
	nop
.L25:
	movq	$0, _TIG_IZ_i38W_argv(%rip)
	nop
.L26:
	movl	$0, _TIG_IZ_i38W_argc(%rip)
	nop
	nop
.L27:
.L28:
#APP
# 137 "mblaszczykowski_algorithms-data-structures-c_5.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-i38W--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_i38W_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_i38W_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_i38W_envp(%rip)
	nop
	movq	$2, -120(%rbp)
.L34:
	cmpq	$2, -120(%rbp)
	je	.L29
	cmpq	$2, -120(%rbp)
	ja	.L37
	cmpq	$0, -120(%rbp)
	je	.L31
	cmpq	$1, -120(%rbp)
	jne	.L37
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	ONP
	movq	$0, -120(%rbp)
	jmp	.L32
.L31:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L35
	jmp	.L36
.L29:
	movq	$1, -120(%rbp)
	jmp	.L32
.L37:
	nop
.L32:
	jmp	.L34
.L36:
	call	__stack_chk_fail@PLT
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.globl	push
	.type	push, @function
push:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$2, -24(%rbp)
.L44:
	cmpq	$2, -24(%rbp)
	je	.L39
	cmpq	$2, -24(%rbp)
	ja	.L46
	cmpq	$0, -24(%rbp)
	je	.L41
	cmpq	$1, -24(%rbp)
	jne	.L46
	jmp	.L45
.L41:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	top(%rip), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, top(%rip)
	movq	$1, -24(%rbp)
	jmp	.L43
.L39:
	movq	$0, -24(%rbp)
	jmp	.L43
.L46:
	nop
.L43:
	jmp	.L44
.L45:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	push, .-push
	.globl	showTop
	.type	showTop, @function
showTop:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L50:
	cmpq	$0, -8(%rbp)
	jne	.L53
	movq	top(%rip), %rax
	movl	(%rax), %eax
	jmp	.L52
.L53:
	nop
	jmp	.L50
.L52:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	showTop, .-showTop
	.section	.rodata
.LC4:
	.string	"%d "
	.text
	.globl	display
	.type	display, @function
display:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L70:
	cmpq	$9, -8(%rbp)
	ja	.L71
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L57(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L57(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L57:
	.long	.L63-.L57
	.long	.L62-.L57
	.long	.L71-.L57
	.long	.L71-.L57
	.long	.L61-.L57
	.long	.L60-.L57
	.long	.L72-.L57
	.long	.L58-.L57
	.long	.L71-.L57
	.long	.L56-.L57
	.text
.L61:
	movq	top(%rip), %rax
	testq	%rax, %rax
	jne	.L64
	movq	$5, -8(%rbp)
	jmp	.L66
.L64:
	movq	$1, -8(%rbp)
	jmp	.L66
.L62:
	movq	top(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L66
.L56:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L66
.L60:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L66
.L63:
	cmpq	$0, -16(%rbp)
	je	.L68
	movq	$9, -8(%rbp)
	jmp	.L66
.L68:
	movq	$7, -8(%rbp)
	jmp	.L66
.L58:
	movl	$10, %edi
	call	putchar@PLT
	movq	$6, -8(%rbp)
	jmp	.L66
.L71:
	nop
.L66:
	jmp	.L70
.L72:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	display, .-display
	.globl	priorytety
	.type	priorytety, @function
priorytety:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$7, -8(%rbp)
.L99:
	cmpq	$11, -8(%rbp)
	ja	.L100
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L76(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L76(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L76:
	.long	.L86-.L76
	.long	.L100-.L76
	.long	.L85-.L76
	.long	.L84-.L76
	.long	.L83-.L76
	.long	.L82-.L76
	.long	.L81-.L76
	.long	.L80-.L76
	.long	.L79-.L76
	.long	.L78-.L76
	.long	.L77-.L76
	.long	.L75-.L76
	.text
.L83:
	cmpb	$45, -20(%rbp)
	jne	.L87
	movq	$11, -8(%rbp)
	jmp	.L89
.L87:
	movq	$10, -8(%rbp)
	jmp	.L89
.L79:
	movl	$2, %eax
	jmp	.L90
.L84:
	cmpb	$47, -20(%rbp)
	jne	.L91
	movq	$8, -8(%rbp)
	jmp	.L89
.L91:
	movq	$5, -8(%rbp)
	jmp	.L89
.L75:
	movl	$1, %eax
	jmp	.L90
.L78:
	cmpb	$43, -20(%rbp)
	jne	.L93
	movq	$0, -8(%rbp)
	jmp	.L89
.L93:
	movq	$4, -8(%rbp)
	jmp	.L89
.L81:
	movl	$3, %eax
	jmp	.L90
.L82:
	cmpb	$42, -20(%rbp)
	jne	.L95
	movq	$2, -8(%rbp)
	jmp	.L89
.L95:
	movq	$9, -8(%rbp)
	jmp	.L89
.L77:
	movl	$-1, %eax
	jmp	.L90
.L86:
	movl	$1, %eax
	jmp	.L90
.L80:
	cmpb	$94, -20(%rbp)
	jne	.L97
	movq	$6, -8(%rbp)
	jmp	.L89
.L97:
	movq	$3, -8(%rbp)
	jmp	.L89
.L85:
	movl	$2, %eax
	jmp	.L90
.L100:
	nop
.L89:
	jmp	.L99
.L90:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	priorytety, .-priorytety
	.globl	ONP
	.type	ONP, @function
ONP:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movq	%rdi, -200(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$13, -128(%rbp)
.L170:
	cmpq	$57, -128(%rbp)
	ja	.L173
	movq	-128(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L104(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L104(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L104:
	.long	.L173-.L104
	.long	.L139-.L104
	.long	.L138-.L104
	.long	.L173-.L104
	.long	.L137-.L104
	.long	.L173-.L104
	.long	.L136-.L104
	.long	.L135-.L104
	.long	.L134-.L104
	.long	.L133-.L104
	.long	.L132-.L104
	.long	.L173-.L104
	.long	.L131-.L104
	.long	.L130-.L104
	.long	.L129-.L104
	.long	.L173-.L104
	.long	.L173-.L104
	.long	.L174-.L104
	.long	.L127-.L104
	.long	.L173-.L104
	.long	.L126-.L104
	.long	.L173-.L104
	.long	.L125-.L104
	.long	.L124-.L104
	.long	.L123-.L104
	.long	.L122-.L104
	.long	.L173-.L104
	.long	.L121-.L104
	.long	.L173-.L104
	.long	.L120-.L104
	.long	.L119-.L104
	.long	.L118-.L104
	.long	.L173-.L104
	.long	.L173-.L104
	.long	.L173-.L104
	.long	.L117-.L104
	.long	.L173-.L104
	.long	.L173-.L104
	.long	.L173-.L104
	.long	.L116-.L104
	.long	.L115-.L104
	.long	.L114-.L104
	.long	.L173-.L104
	.long	.L173-.L104
	.long	.L113-.L104
	.long	.L112-.L104
	.long	.L111-.L104
	.long	.L173-.L104
	.long	.L110-.L104
	.long	.L109-.L104
	.long	.L173-.L104
	.long	.L108-.L104
	.long	.L173-.L104
	.long	.L107-.L104
	.long	.L106-.L104
	.long	.L105-.L104
	.long	.L173-.L104
	.long	.L103-.L104
	.text
.L127:
	movzbl	-181(%rbp), %eax
	movb	%al, -120(%rbp)
	movb	$0, -119(%rbp)
	leaq	-120(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$4, -128(%rbp)
	jmp	.L140
.L122:
	call	showTop
	movl	%eax, -172(%rbp)
	movq	$46, -128(%rbp)
	jmp	.L140
.L109:
	movl	-164(%rbp), %eax
	cmpl	-160(%rbp), %eax
	jg	.L141
	movq	$29, -128(%rbp)
	jmp	.L140
.L141:
	movq	$2, -128(%rbp)
	jmp	.L140
.L137:
	addl	$1, -176(%rbp)
	movq	$53, -128(%rbp)
	jmp	.L140
.L119:
	movzbl	-181(%rbp), %eax
	movb	%al, -120(%rbp)
	movb	$0, -119(%rbp)
	leaq	-120(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$4, -128(%rbp)
	jmp	.L140
.L129:
	cmpb	$90, -181(%rbp)
	jg	.L143
	movq	$30, -128(%rbp)
	jmp	.L140
.L143:
	movq	$51, -128(%rbp)
	jmp	.L140
.L118:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -128(%rbp)
	jmp	.L140
.L131:
	call	showTop
	movl	%eax, -140(%rbp)
	movl	-140(%rbp), %eax
	movb	%al, -114(%rbp)
	movb	$0, -113(%rbp)
	leaq	-114(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	call	pop
	movq	$48, -128(%rbp)
	jmp	.L140
.L134:
	movb	$0, -112(%rbp)
	movl	$1, -180(%rbp)
	movq	$24, -128(%rbp)
	jmp	.L140
.L112:
	movl	$0, -176(%rbp)
	movq	$53, -128(%rbp)
	jmp	.L140
.L106:
	cmpb	$96, -181(%rbp)
	jle	.L145
	movq	$35, -128(%rbp)
	jmp	.L140
.L145:
	movq	$39, -128(%rbp)
	jmp	.L140
.L139:
	movzbl	-181(%rbp), %eax
	movb	%al, -120(%rbp)
	movb	$0, -119(%rbp)
	leaq	-120(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$4, -128(%rbp)
	jmp	.L140
.L124:
	cmpl	$0, -156(%rbp)
	je	.L147
	movq	$31, -128(%rbp)
	jmp	.L140
.L147:
	movq	$12, -128(%rbp)
	jmp	.L140
.L123:
	cmpl	$99, -180(%rbp)
	jbe	.L149
	movq	$45, -128(%rbp)
	jmp	.L140
.L149:
	movq	$22, -128(%rbp)
	jmp	.L140
.L103:
	call	empty
	movl	%eax, -168(%rbp)
	movq	$6, -128(%rbp)
	jmp	.L140
.L133:
	cmpb	$40, -181(%rbp)
	jne	.L151
	movq	$44, -128(%rbp)
	jmp	.L140
.L151:
	movq	$40, -128(%rbp)
	jmp	.L140
.L130:
	movq	$8, -128(%rbp)
	jmp	.L140
.L108:
	cmpb	$47, -181(%rbp)
	jle	.L153
	movq	$7, -128(%rbp)
	jmp	.L140
.L153:
	movq	$9, -128(%rbp)
	jmp	.L140
.L115:
	cmpb	$41, -181(%rbp)
	jne	.L156
	movq	$25, -128(%rbp)
	jmp	.L140
.L156:
	movq	$57, -128(%rbp)
	jmp	.L140
.L105:
	call	pop
	movq	$4, -128(%rbp)
	jmp	.L140
.L136:
	cmpl	$0, -168(%rbp)
	je	.L158
	movq	$2, -128(%rbp)
	jmp	.L140
.L158:
	movq	$41, -128(%rbp)
	jmp	.L140
.L121:
	movl	-176(%rbp), %eax
	cltq
	cmpq	%rax, -136(%rbp)
	jbe	.L160
	movq	$10, -128(%rbp)
	jmp	.L140
.L160:
	movq	$48, -128(%rbp)
	jmp	.L140
.L110:
	call	empty
	movl	%eax, -156(%rbp)
	movq	$23, -128(%rbp)
	jmp	.L140
.L125:
	movl	-180(%rbp), %eax
	movb	$0, -112(%rbp,%rax)
	addl	$1, -180(%rbp)
	movq	$24, -128(%rbp)
	jmp	.L140
.L107:
	movq	-200(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -136(%rbp)
	movq	$27, -128(%rbp)
	jmp	.L140
.L113:
	movl	$40, %edi
	call	push
	movq	$4, -128(%rbp)
	jmp	.L140
.L114:
	movl	-176(%rbp), %eax
	movslq	%eax, %rdx
	movq	-200(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	priorytety
	movl	%eax, -164(%rbp)
	call	showTop
	movl	%eax, -152(%rbp)
	movl	-152(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	priorytety
	movl	%eax, -160(%rbp)
	movq	$49, -128(%rbp)
	jmp	.L140
.L132:
	movl	-176(%rbp), %eax
	movslq	%eax, %rdx
	movq	-200(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -181(%rbp)
	movq	$54, -128(%rbp)
	jmp	.L140
.L111:
	cmpl	$40, -172(%rbp)
	je	.L162
	movq	$20, -128(%rbp)
	jmp	.L140
.L162:
	movq	$55, -128(%rbp)
	jmp	.L140
.L116:
	cmpb	$64, -181(%rbp)
	jle	.L164
	movq	$14, -128(%rbp)
	jmp	.L140
.L164:
	movq	$51, -128(%rbp)
	jmp	.L140
.L135:
	cmpb	$57, -181(%rbp)
	jg	.L166
	movq	$1, -128(%rbp)
	jmp	.L140
.L166:
	movq	$9, -128(%rbp)
	jmp	.L140
.L117:
	cmpb	$122, -181(%rbp)
	jg	.L168
	movq	$18, -128(%rbp)
	jmp	.L140
.L168:
	movq	$39, -128(%rbp)
	jmp	.L140
.L120:
	call	showTop
	movl	%eax, -148(%rbp)
	movl	-148(%rbp), %eax
	movb	%al, -116(%rbp)
	movb	$0, -115(%rbp)
	leaq	-116(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	call	pop
	movq	$57, -128(%rbp)
	jmp	.L140
.L138:
	movsbl	-181(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$4, -128(%rbp)
	jmp	.L140
.L126:
	call	showTop
	movl	%eax, -144(%rbp)
	movl	-144(%rbp), %eax
	movb	%al, -118(%rbp)
	movb	$0, -117(%rbp)
	leaq	-118(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	call	pop
	movq	$25, -128(%rbp)
	jmp	.L140
.L173:
	nop
.L140:
	jmp	.L170
.L174:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L172
	call	__stack_chk_fail@PLT
.L172:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	ONP, .-ONP
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
