	.file	"zetavg_201411-OSHW1_b_flatten.c"
	.text
	.globl	_TIG_IZ_qDMs_argc
	.bss
	.align 4
	.type	_TIG_IZ_qDMs_argc, @object
	.size	_TIG_IZ_qDMs_argc, 4
_TIG_IZ_qDMs_argc:
	.zero	4
	.globl	wo_hashtab
	.align 32
	.type	wo_hashtab, @object
	.size	wo_hashtab, 1024
wo_hashtab:
	.zero	1024
	.globl	_TIG_IZ_qDMs_argv
	.align 8
	.type	_TIG_IZ_qDMs_argv, @object
	.size	_TIG_IZ_qDMs_argv, 8
_TIG_IZ_qDMs_argv:
	.zero	8
	.globl	_TIG_IZ_qDMs_envp
	.align 8
	.type	_TIG_IZ_qDMs_envp, @object
	.size	_TIG_IZ_qDMs_envp, 8
_TIG_IZ_qDMs_envp:
	.zero	8
	.text
	.globl	wo_lookup
	.type	wo_lookup, @function
wo_lookup:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$9, -8(%rbp)
.L18:
	cmpq	$9, -8(%rbp)
	ja	.L19
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L19-.L4
	.long	.L19-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	cmpl	$0, -24(%rbp)
	jne	.L12
	movq	$7, -8(%rbp)
	jmp	.L14
.L12:
	movq	$5, -8(%rbp)
	jmp	.L14
.L10:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -24(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L14
.L3:
	movq	$6, -8(%rbp)
	jmp	.L14
.L7:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	hash
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	leaq	0(,%rax,8), %rdx
	leaq	wo_hashtab(%rip), %rax
	movq	(%rdx,%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L14
.L8:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L14
.L11:
	cmpq	$0, -16(%rbp)
	je	.L15
	movq	$1, -8(%rbp)
	jmp	.L14
.L15:
	movq	$2, -8(%rbp)
	jmp	.L14
.L6:
	movq	-16(%rbp), %rax
	jmp	.L17
.L9:
	movl	$0, %eax
	jmp	.L17
.L19:
	nop
.L14:
	jmp	.L18
.L17:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	wo_lookup, .-wo_lookup
	.globl	hash
	.type	hash, @function
hash:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L29:
	cmpq	$3, -8(%rbp)
	je	.L21
	cmpq	$3, -8(%rbp)
	ja	.L31
	cmpq	$2, -8(%rbp)
	je	.L23
	cmpq	$2, -8(%rbp)
	ja	.L31
	cmpq	$0, -8(%rbp)
	je	.L24
	cmpq	$1, -8(%rbp)
	jne	.L31
	movl	-12(%rbp), %eax
	andl	$127, %eax
	jmp	.L30
.L21:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %ecx
	movl	-12(%rbp), %edx
	movl	%edx, %eax
	sall	$5, %eax
	subl	%edx, %eax
	addl	%ecx, %eax
	movl	%eax, -12(%rbp)
	addq	$1, -24(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L26
.L24:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L27
	movq	$3, -8(%rbp)
	jmp	.L26
.L27:
	movq	$1, -8(%rbp)
	jmp	.L26
.L23:
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L26
.L31:
	nop
.L26:
	jmp	.L29
.L30:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	hash, .-hash
	.globl	wo_get
	.type	wo_get, @function
wo_get:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$2, -16(%rbp)
.L44:
	cmpq	$5, -16(%rbp)
	ja	.L45
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L35(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L35(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L35:
	.long	.L39-.L35
	.long	.L38-.L35
	.long	.L37-.L35
	.long	.L45-.L35
	.long	.L36-.L35
	.long	.L34-.L35
	.text
.L36:
	cmpq	$0, -24(%rbp)
	jne	.L40
	movq	$1, -16(%rbp)
	jmp	.L42
.L40:
	movq	$5, -16(%rbp)
	jmp	.L42
.L38:
	movl	$0, %eax
	jmp	.L43
.L34:
	movq	-24(%rbp), %rax
	movl	16(%rax), %eax
	jmp	.L43
.L39:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	wo_lookup
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L42
.L37:
	movq	$0, -16(%rbp)
	jmp	.L42
.L45:
	nop
.L42:
	jmp	.L44
.L43:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	wo_get, .-wo_get
	.section	.rodata
.LC0:
	.string	"%s %d\n"
	.text
	.globl	wo_print_all
	.type	wo_print_all, @function
wo_print_all:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$11, -8(%rbp)
.L62:
	cmpq	$11, -8(%rbp)
	ja	.L63
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L49(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L49(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L49:
	.long	.L55-.L49
	.long	.L54-.L49
	.long	.L63-.L49
	.long	.L53-.L49
	.long	.L63-.L49
	.long	.L63-.L49
	.long	.L63-.L49
	.long	.L64-.L49
	.long	.L51-.L49
	.long	.L50-.L49
	.long	.L63-.L49
	.long	.L48-.L49
	.text
.L51:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	wo_hashtab(%rip), %rax
	movq	(%rdx,%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L56
.L54:
	addl	$1, -20(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L56
.L53:
	cmpl	$127, -20(%rbp)
	jg	.L57
	movq	$8, -8(%rbp)
	jmp	.L56
.L57:
	movq	$7, -8(%rbp)
	jmp	.L56
.L48:
	movl	$0, -20(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L56
.L50:
	movq	-16(%rbp), %rax
	movl	16(%rax), %edx
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L56
.L55:
	cmpq	$0, -16(%rbp)
	je	.L59
	movq	$9, -8(%rbp)
	jmp	.L56
.L59:
	movq	$1, -8(%rbp)
	jmp	.L56
.L63:
	nop
.L56:
	jmp	.L62
.L64:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	wo_print_all, .-wo_print_all
	.globl	wo_plus
	.type	wo_plus, @function
wo_plus:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$3, -24(%rbp)
.L79:
	cmpq	$8, -24(%rbp)
	ja	.L80
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L68(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L68(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L68:
	.long	.L74-.L68
	.long	.L73-.L68
	.long	.L80-.L68
	.long	.L72-.L68
	.long	.L71-.L68
	.long	.L70-.L68
	.long	.L80-.L68
	.long	.L69-.L68
	.long	.L67-.L68
	.text
.L71:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	hash
	movl	%eax, -40(%rbp)
	movl	-40(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-32(%rbp), %rax
	movl	$1, 16(%rax)
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	wo_hashtab(%rip), %rax
	movq	(%rdx,%rax), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, (%rax)
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rcx
	leaq	wo_hashtab(%rip), %rdx
	movq	-32(%rbp), %rax
	movq	%rax, (%rcx,%rdx)
	movq	$0, -24(%rbp)
	jmp	.L75
.L67:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	wo_lookup
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$5, -24(%rbp)
	jmp	.L75
.L73:
	movl	-44(%rbp), %eax
	jmp	.L76
.L72:
	movq	$8, -24(%rbp)
	jmp	.L75
.L70:
	cmpq	$0, -32(%rbp)
	jne	.L77
	movq	$4, -24(%rbp)
	jmp	.L75
.L77:
	movq	$7, -24(%rbp)
	jmp	.L75
.L74:
	movl	$1, %eax
	jmp	.L76
.L69:
	movq	-32(%rbp), %rax
	movl	16(%rax), %eax
	movl	%eax, -44(%rbp)
	movq	-32(%rbp), %rax
	movl	16(%rax), %eax
	leal	1(%rax), %edx
	movq	-32(%rbp), %rax
	movl	%edx, 16(%rax)
	movq	$1, -24(%rbp)
	jmp	.L75
.L80:
	nop
.L75:
	jmp	.L79
.L76:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	wo_plus, .-wo_plus
	.section	.rodata
.LC1:
	.string	"ab_shm_8617ffee"
.LC2:
	.string	"shm failed"
.LC3:
	.string	""
.LC4:
	.string	"no"
.LC5:
	.string	"ptr failed"
.LC6:
	.string	"yes"
	.text
	.globl	main
	.type	main, @function
main:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$0, wo_hashtab(%rip)
	movq	$0, 8+wo_hashtab(%rip)
	movq	$0, 16+wo_hashtab(%rip)
	movq	$0, 24+wo_hashtab(%rip)
	movq	$0, 32+wo_hashtab(%rip)
	movq	$0, 40+wo_hashtab(%rip)
	movq	$0, 48+wo_hashtab(%rip)
	movq	$0, 56+wo_hashtab(%rip)
	movq	$0, 64+wo_hashtab(%rip)
	movq	$0, 72+wo_hashtab(%rip)
	movq	$0, 80+wo_hashtab(%rip)
	movq	$0, 88+wo_hashtab(%rip)
	movq	$0, 96+wo_hashtab(%rip)
	movq	$0, 104+wo_hashtab(%rip)
	movq	$0, 112+wo_hashtab(%rip)
	movq	$0, 120+wo_hashtab(%rip)
	movq	$0, 128+wo_hashtab(%rip)
	movq	$0, 136+wo_hashtab(%rip)
	movq	$0, 144+wo_hashtab(%rip)
	movq	$0, 152+wo_hashtab(%rip)
	movq	$0, 160+wo_hashtab(%rip)
	movq	$0, 168+wo_hashtab(%rip)
	movq	$0, 176+wo_hashtab(%rip)
	movq	$0, 184+wo_hashtab(%rip)
	movq	$0, 192+wo_hashtab(%rip)
	movq	$0, 200+wo_hashtab(%rip)
	movq	$0, 208+wo_hashtab(%rip)
	movq	$0, 216+wo_hashtab(%rip)
	movq	$0, 224+wo_hashtab(%rip)
	movq	$0, 232+wo_hashtab(%rip)
	movq	$0, 240+wo_hashtab(%rip)
	movq	$0, 248+wo_hashtab(%rip)
	movq	$0, 256+wo_hashtab(%rip)
	movq	$0, 264+wo_hashtab(%rip)
	movq	$0, 272+wo_hashtab(%rip)
	movq	$0, 280+wo_hashtab(%rip)
	movq	$0, 288+wo_hashtab(%rip)
	movq	$0, 296+wo_hashtab(%rip)
	movq	$0, 304+wo_hashtab(%rip)
	movq	$0, 312+wo_hashtab(%rip)
	movq	$0, 320+wo_hashtab(%rip)
	movq	$0, 328+wo_hashtab(%rip)
	movq	$0, 336+wo_hashtab(%rip)
	movq	$0, 344+wo_hashtab(%rip)
	movq	$0, 352+wo_hashtab(%rip)
	movq	$0, 360+wo_hashtab(%rip)
	movq	$0, 368+wo_hashtab(%rip)
	movq	$0, 376+wo_hashtab(%rip)
	movq	$0, 384+wo_hashtab(%rip)
	movq	$0, 392+wo_hashtab(%rip)
	movq	$0, 400+wo_hashtab(%rip)
	movq	$0, 408+wo_hashtab(%rip)
	movq	$0, 416+wo_hashtab(%rip)
	movq	$0, 424+wo_hashtab(%rip)
	movq	$0, 432+wo_hashtab(%rip)
	movq	$0, 440+wo_hashtab(%rip)
	movq	$0, 448+wo_hashtab(%rip)
	movq	$0, 456+wo_hashtab(%rip)
	movq	$0, 464+wo_hashtab(%rip)
	movq	$0, 472+wo_hashtab(%rip)
	movq	$0, 480+wo_hashtab(%rip)
	movq	$0, 488+wo_hashtab(%rip)
	movq	$0, 496+wo_hashtab(%rip)
	movq	$0, 504+wo_hashtab(%rip)
	movq	$0, 512+wo_hashtab(%rip)
	movq	$0, 520+wo_hashtab(%rip)
	movq	$0, 528+wo_hashtab(%rip)
	movq	$0, 536+wo_hashtab(%rip)
	movq	$0, 544+wo_hashtab(%rip)
	movq	$0, 552+wo_hashtab(%rip)
	movq	$0, 560+wo_hashtab(%rip)
	movq	$0, 568+wo_hashtab(%rip)
	movq	$0, 576+wo_hashtab(%rip)
	movq	$0, 584+wo_hashtab(%rip)
	movq	$0, 592+wo_hashtab(%rip)
	movq	$0, 600+wo_hashtab(%rip)
	movq	$0, 608+wo_hashtab(%rip)
	movq	$0, 616+wo_hashtab(%rip)
	movq	$0, 624+wo_hashtab(%rip)
	movq	$0, 632+wo_hashtab(%rip)
	movq	$0, 640+wo_hashtab(%rip)
	movq	$0, 648+wo_hashtab(%rip)
	movq	$0, 656+wo_hashtab(%rip)
	movq	$0, 664+wo_hashtab(%rip)
	movq	$0, 672+wo_hashtab(%rip)
	movq	$0, 680+wo_hashtab(%rip)
	movq	$0, 688+wo_hashtab(%rip)
	movq	$0, 696+wo_hashtab(%rip)
	movq	$0, 704+wo_hashtab(%rip)
	movq	$0, 712+wo_hashtab(%rip)
	movq	$0, 720+wo_hashtab(%rip)
	movq	$0, 728+wo_hashtab(%rip)
	movq	$0, 736+wo_hashtab(%rip)
	movq	$0, 744+wo_hashtab(%rip)
	movq	$0, 752+wo_hashtab(%rip)
	movq	$0, 760+wo_hashtab(%rip)
	movq	$0, 768+wo_hashtab(%rip)
	movq	$0, 776+wo_hashtab(%rip)
	movq	$0, 784+wo_hashtab(%rip)
	movq	$0, 792+wo_hashtab(%rip)
	movq	$0, 800+wo_hashtab(%rip)
	movq	$0, 808+wo_hashtab(%rip)
	movq	$0, 816+wo_hashtab(%rip)
	movq	$0, 824+wo_hashtab(%rip)
	movq	$0, 832+wo_hashtab(%rip)
	movq	$0, 840+wo_hashtab(%rip)
	movq	$0, 848+wo_hashtab(%rip)
	movq	$0, 856+wo_hashtab(%rip)
	movq	$0, 864+wo_hashtab(%rip)
	movq	$0, 872+wo_hashtab(%rip)
	movq	$0, 880+wo_hashtab(%rip)
	movq	$0, 888+wo_hashtab(%rip)
	movq	$0, 896+wo_hashtab(%rip)
	movq	$0, 904+wo_hashtab(%rip)
	movq	$0, 912+wo_hashtab(%rip)
	movq	$0, 920+wo_hashtab(%rip)
	movq	$0, 928+wo_hashtab(%rip)
	movq	$0, 936+wo_hashtab(%rip)
	movq	$0, 944+wo_hashtab(%rip)
	movq	$0, 952+wo_hashtab(%rip)
	movq	$0, 960+wo_hashtab(%rip)
	movq	$0, 968+wo_hashtab(%rip)
	movq	$0, 976+wo_hashtab(%rip)
	movq	$0, 984+wo_hashtab(%rip)
	movq	$0, 992+wo_hashtab(%rip)
	movq	$0, 1000+wo_hashtab(%rip)
	movq	$0, 1008+wo_hashtab(%rip)
	movq	$0, 1016+wo_hashtab(%rip)
	nop
.L82:
	movq	$0, _TIG_IZ_qDMs_envp(%rip)
	nop
.L83:
	movq	$0, _TIG_IZ_qDMs_argv(%rip)
	nop
.L84:
	movl	$0, _TIG_IZ_qDMs_argc(%rip)
	nop
	nop
.L85:
.L86:
#APP
# 248 "zetavg_201411-OSHW1_b.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qDMs--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_qDMs_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_qDMs_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_qDMs_envp(%rip)
	nop
	movq	$7, -24(%rbp)
.L123:
	cmpq	$24, -24(%rbp)
	ja	.L124
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L89(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L89(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L89:
	.long	.L108-.L89
	.long	.L124-.L89
	.long	.L107-.L89
	.long	.L106-.L89
	.long	.L105-.L89
	.long	.L104-.L89
	.long	.L103-.L89
	.long	.L102-.L89
	.long	.L101-.L89
	.long	.L100-.L89
	.long	.L99-.L89
	.long	.L98-.L89
	.long	.L97-.L89
	.long	.L124-.L89
	.long	.L96-.L89
	.long	.L124-.L89
	.long	.L95-.L89
	.long	.L94-.L89
	.long	.L124-.L89
	.long	.L93-.L89
	.long	.L124-.L89
	.long	.L92-.L89
	.long	.L91-.L89
	.long	.L90-.L89
	.long	.L88-.L89
	.text
.L105:
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	wo_get
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, -52(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L109
.L96:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$36, %al
	jne	.L110
	movq	$2, -24(%rbp)
	jmp	.L109
.L110:
	movq	$21, -24(%rbp)
	jmp	.L109
.L97:
	movl	$511, %edx
	movl	$66, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	shm_open@PLT
	movl	%eax, -48(%rbp)
	movl	-48(%rbp), %eax
	movl	%eax, -56(%rbp)
	movq	$9, -24(%rbp)
	jmp	.L109
.L101:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -24(%rbp)
	jmp	.L109
.L90:
	movq	-40(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movl	$1000, %edi
	call	usleep@PLT
	movq	$19, -24(%rbp)
	jmp	.L109
.L106:
	cmpl	$0, -52(%rbp)
	jg	.L112
	movq	$22, -24(%rbp)
	jmp	.L109
.L112:
	movq	$0, -24(%rbp)
	jmp	.L109
.L95:
	movl	$-1, %eax
	jmp	.L114
.L88:
	movl	-56(%rbp), %eax
	movl	$1024, %esi
	movl	%eax, %edi
	call	ftruncate@PLT
	movl	-56(%rbp), %eax
	movl	$0, %r9d
	movl	%eax, %r8d
	movl	$1, %ecx
	movl	$3, %edx
	movl	$1024, %esi
	movl	$0, %edi
	call	mmap@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$17, -24(%rbp)
	jmp	.L109
.L92:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	wo_plus
	movq	$23, -24(%rbp)
	jmp	.L109
.L98:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$37, %al
	jne	.L115
	movq	$4, -24(%rbp)
	jmp	.L109
.L115:
	movq	$14, -24(%rbp)
	jmp	.L109
.L100:
	cmpl	$-1, -56(%rbp)
	jne	.L117
	movq	$8, -24(%rbp)
	jmp	.L109
.L117:
	movq	$24, -24(%rbp)
	jmp	.L109
.L93:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -32(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L109
.L94:
	cmpq	$0, -40(%rbp)
	jne	.L119
	movq	$5, -24(%rbp)
	jmp	.L109
.L119:
	movq	$19, -24(%rbp)
	jmp	.L109
.L103:
	movl	$-1, %eax
	jmp	.L114
.L91:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$23, -24(%rbp)
	jmp	.L109
.L104:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$16, -24(%rbp)
	jmp	.L109
.L99:
	cmpq	$0, -32(%rbp)
	je	.L121
	movq	$11, -24(%rbp)
	jmp	.L109
.L121:
	movq	$23, -24(%rbp)
	jmp	.L109
.L108:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$23, -24(%rbp)
	jmp	.L109
.L102:
	movq	$12, -24(%rbp)
	jmp	.L109
.L107:
	call	wo_print_all
	movq	$23, -24(%rbp)
	jmp	.L109
.L124:
	nop
.L109:
	jmp	.L123
.L114:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	main, .-main
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
