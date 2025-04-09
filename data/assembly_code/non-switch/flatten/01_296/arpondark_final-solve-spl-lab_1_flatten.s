	.file	"arpondark_final-solve-spl-lab_1_flatten.c"
	.text
	.globl	_TIG_IZ_7mCp_argv
	.bss
	.align 8
	.type	_TIG_IZ_7mCp_argv, @object
	.size	_TIG_IZ_7mCp_argv, 8
_TIG_IZ_7mCp_argv:
	.zero	8
	.globl	_TIG_IZ_7mCp_envp
	.align 8
	.type	_TIG_IZ_7mCp_envp, @object
	.size	_TIG_IZ_7mCp_envp, 8
_TIG_IZ_7mCp_envp:
	.zero	8
	.globl	_TIG_IZ_7mCp_argc
	.align 4
	.type	_TIG_IZ_7mCp_argc, @object
	.size	_TIG_IZ_7mCp_argc, 4
_TIG_IZ_7mCp_argc:
	.zero	4
	.text
	.globl	isDragon
	.type	isDragon, @function
isDragon:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -16(%rbp)
.L25:
	cmpq	$11, -16(%rbp)
	ja	.L28
	movq	-16(%rbp), %rax
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
	.long	.L14-.L4
	.long	.L28-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L11:
	movl	$0, %eax
	jmp	.L26
.L7:
	leaq	-32(%rbp), %rdx
	leaq	-36(%rbp), %rcx
	movq	-56(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	getMiddleWord
	movq	$3, -16(%rbp)
	jmp	.L16
.L12:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$63, %al
	jne	.L17
	movq	$9, -16(%rbp)
	jmp	.L16
.L17:
	movq	$2, -16(%rbp)
	jmp	.L16
.L3:
	movl	-36(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -28(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L16
.L6:
	movl	-32(%rbp), %eax
	cltq
	leaq	-1(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$35, %al
	jne	.L19
	movq	$11, -16(%rbp)
	jmp	.L16
.L19:
	movq	$2, -16(%rbp)
	jmp	.L16
.L9:
	addl	$1, -28(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L16
.L10:
	movl	$1, %eax
	jmp	.L26
.L5:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	movslq	%eax, %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L21
	movq	$6, -16(%rbp)
	jmp	.L16
.L21:
	movq	$4, -16(%rbp)
	jmp	.L16
.L14:
	movl	-32(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -28(%rbp)
	jge	.L23
	movq	$7, -16(%rbp)
	jmp	.L16
.L23:
	movq	$5, -16(%rbp)
	jmp	.L16
.L8:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L16
.L13:
	movl	$0, %eax
	jmp	.L26
.L28:
	nop
.L16:
	jmp	.L25
.L26:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L27
	call	__stack_chk_fail@PLT
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	isDragon, .-isDragon
	.globl	countWords
	.type	countWords, @function
countWords:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$7, -8(%rbp)
.L49:
	cmpq	$11, -8(%rbp)
	ja	.L51
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L40-.L32
	.long	.L39-.L32
	.long	.L38-.L32
	.long	.L51-.L32
	.long	.L37-.L32
	.long	.L36-.L32
	.long	.L35-.L32
	.long	.L34-.L32
	.long	.L33-.L32
	.long	.L51-.L32
	.long	.L51-.L32
	.long	.L31-.L32
	.text
.L37:
	movl	-16(%rbp), %eax
	jmp	.L50
.L33:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L42
	movq	$2, -8(%rbp)
	jmp	.L44
.L42:
	movq	$4, -8(%rbp)
	jmp	.L44
.L39:
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L44
.L31:
	movl	$1, -12(%rbp)
	addl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L44
.L35:
	movl	$0, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L44
.L36:
	cmpl	$0, -12(%rbp)
	jne	.L45
	movq	$11, -8(%rbp)
	jmp	.L44
.L45:
	movq	$0, -8(%rbp)
	jmp	.L44
.L40:
	addq	$1, -24(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L44
.L34:
	movq	$6, -8(%rbp)
	jmp	.L44
.L38:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	jne	.L47
	movq	$1, -8(%rbp)
	jmp	.L44
.L47:
	movq	$5, -8(%rbp)
	jmp	.L44
.L51:
	nop
.L44:
	jmp	.L49
.L50:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	countWords, .-countWords
	.globl	getMiddleWord
	.type	getMiddleWord, @function
getMiddleWord:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$56, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$16, -8(%rbp)
.L90:
	cmpq	$28, -8(%rbp)
	ja	.L91
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L55(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L55(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L55:
	.long	.L91-.L55
	.long	.L71-.L55
	.long	.L70-.L55
	.long	.L69-.L55
	.long	.L68-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L67-.L55
	.long	.L66-.L55
	.long	.L92-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L64-.L55
	.long	.L63-.L55
	.long	.L62-.L55
	.long	.L61-.L55
	.long	.L60-.L55
	.long	.L59-.L55
	.long	.L58-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L91-.L55
	.long	.L57-.L55
	.long	.L56-.L55
	.long	.L54-.L55
	.text
.L60:
	movl	-24(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jne	.L72
	movq	$17, -8(%rbp)
	jmp	.L74
.L72:
	movq	$2, -8(%rbp)
	jmp	.L74
.L68:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L75
	movq	$7, -8(%rbp)
	jmp	.L74
.L75:
	movq	$9, -8(%rbp)
	jmp	.L74
.L64:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	countWords
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	addl	$1, %eax
	movl	%eax, -28(%rbp)
	movl	$0, -24(%rbp)
	movl	$0, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L74
.L63:
	addl	$1, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L74
.L66:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L77
	movq	$15, -8(%rbp)
	jmp	.L74
.L77:
	movq	$28, -8(%rbp)
	jmp	.L74
.L71:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L79
	movq	$8, -8(%rbp)
	jmp	.L74
.L79:
	movq	$28, -8(%rbp)
	jmp	.L74
.L69:
	addl	$1, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L74
.L62:
	movq	$14, -8(%rbp)
	jmp	.L74
.L57:
	addl	$1, -20(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L74
.L59:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L82
	movq	$3, -8(%rbp)
	jmp	.L74
.L82:
	movq	$4, -8(%rbp)
	jmp	.L74
.L61:
	movq	-48(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L74
.L56:
	addl	$1, -24(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L74
.L54:
	movq	-56(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, (%rax)
	movq	$9, -8(%rbp)
	jmp	.L74
.L67:
	movl	-24(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L84
	movq	$20, -8(%rbp)
	jmp	.L74
.L84:
	movq	$9, -8(%rbp)
	jmp	.L74
.L70:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L86
	movq	$19, -8(%rbp)
	jmp	.L74
.L86:
	movq	$4, -8(%rbp)
	jmp	.L74
.L58:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	jne	.L88
	movq	$26, -8(%rbp)
	jmp	.L74
.L88:
	movq	$27, -8(%rbp)
	jmp	.L74
.L91:
	nop
.L74:
	jmp	.L90
.L92:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	getMiddleWord, .-getMiddleWord
	.section	.rodata
.LC0:
	.string	"Enter a string: "
.LC1:
	.string	"No, it's not a Dragon String."
.LC2:
	.string	"Yes, it's a Dragon String."
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, _TIG_IZ_7mCp_envp(%rip)
	nop
.L94:
	movq	$0, _TIG_IZ_7mCp_argv(%rip)
	nop
.L95:
	movl	$0, _TIG_IZ_7mCp_argc(%rip)
	nop
	nop
.L96:
.L97:
#APP
# 114 "arpondark_final-solve-spl-lab_1.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-7mCp--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_7mCp_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_7mCp_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_7mCp_envp(%rip)
	nop
	movq	$1, -120(%rbp)
.L110:
	cmpq	$5, -120(%rbp)
	ja	.L113
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L100(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L100(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L100:
	.long	.L105-.L100
	.long	.L104-.L100
	.long	.L103-.L100
	.long	.L102-.L100
	.long	.L101-.L100
	.long	.L99-.L100
	.text
.L101:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	gets@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	isDragon
	movl	%eax, -124(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L106
.L104:
	movq	$4, -120(%rbp)
	jmp	.L106
.L102:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L111
	jmp	.L112
.L99:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -120(%rbp)
	jmp	.L106
.L105:
	cmpl	$0, -124(%rbp)
	je	.L108
	movq	$2, -120(%rbp)
	jmp	.L106
.L108:
	movq	$5, -120(%rbp)
	jmp	.L106
.L103:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -120(%rbp)
	jmp	.L106
.L113:
	nop
.L106:
	jmp	.L110
.L112:
	call	__stack_chk_fail@PLT
.L111:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
