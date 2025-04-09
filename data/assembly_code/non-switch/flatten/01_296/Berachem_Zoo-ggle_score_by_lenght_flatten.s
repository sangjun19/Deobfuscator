	.file	"Berachem_Zoo-ggle_score_by_lenght_flatten.c"
	.text
	.globl	_TIG_IZ_HkCk_envp
	.bss
	.align 8
	.type	_TIG_IZ_HkCk_envp, @object
	.size	_TIG_IZ_HkCk_envp, 8
_TIG_IZ_HkCk_envp:
	.zero	8
	.globl	_TIG_IZ_HkCk_argv
	.align 8
	.type	_TIG_IZ_HkCk_argv, @object
	.size	_TIG_IZ_HkCk_argv, 8
_TIG_IZ_HkCk_argv:
	.zero	8
	.globl	_TIG_IZ_HkCk_argc
	.align 4
	.type	_TIG_IZ_HkCk_argc, @object
	.size	_TIG_IZ_HkCk_argc, 4
_TIG_IZ_HkCk_argc:
	.zero	4
	.text
	.globl	countOccurences
	.type	countOccurences, @function
countOccurences:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, %eax
	movb	%al, -44(%rbp)
	movq	$2, -8(%rbp)
.L18:
	cmpq	$7, -8(%rbp)
	ja	.L20
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
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movl	-24(%rbp), %eax
	jmp	.L19
.L10:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, -44(%rbp)
	jne	.L13
	movq	$0, -8(%rbp)
	jmp	.L15
.L13:
	movq	$5, -8(%rbp)
	jmp	.L15
.L8:
	movl	$0, -24(%rbp)
	movl	$0, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L15
.L5:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L15
.L6:
	addl	$1, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L15
.L11:
	addl	$1, -24(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L15
.L3:
	movl	-20(%rbp), %eax
	cltq
	cmpq	%rax, -16(%rbp)
	jbe	.L16
	movq	$1, -8(%rbp)
	jmp	.L15
.L16:
	movq	$4, -8(%rbp)
	jmp	.L15
.L9:
	movq	$3, -8(%rbp)
	jmp	.L15
.L20:
	nop
.L15:
	jmp	.L18
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	countOccurences, .-countOccurences
	.globl	scoreByWord
	.type	scoreByWord, @function
scoreByWord:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$9, -16(%rbp)
.L57:
	cmpq	$16, -16(%rbp)
	ja	.L59
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L40-.L24
	.long	.L39-.L24
	.long	.L38-.L24
	.long	.L37-.L24
	.long	.L36-.L24
	.long	.L35-.L24
	.long	.L34-.L24
	.long	.L33-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L36:
	cmpl	$3, -24(%rbp)
	jne	.L41
	movq	$10, -16(%rbp)
	jmp	.L43
.L41:
	movq	$16, -16(%rbp)
	jmp	.L43
.L26:
	cmpl	$2, -24(%rbp)
	jne	.L44
	movq	$1, -16(%rbp)
	jmp	.L43
.L44:
	movq	$4, -16(%rbp)
	jmp	.L43
.L25:
	movl	$2, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L43
.L28:
	movl	$0, -28(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movl	$42, %esi
	movq	%rax, %rdi
	call	countOccurences
	movl	%eax, -20(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -24(%rbp)
	movq	$14, -16(%rbp)
	jmp	.L43
.L32:
	cmpl	$5, -24(%rbp)
	jne	.L46
	movq	$15, -16(%rbp)
	jmp	.L43
.L46:
	movq	$2, -16(%rbp)
	jmp	.L43
.L39:
	movl	$1, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L43
.L37:
	movl	$1, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L43
.L23:
	cmpl	$4, -24(%rbp)
	jne	.L48
	movq	$3, -16(%rbp)
	jmp	.L43
.L48:
	movq	$8, -16(%rbp)
	jmp	.L43
.L29:
	cmpl	$7, -24(%rbp)
	jne	.L50
	movq	$0, -16(%rbp)
	jmp	.L43
.L50:
	movq	$7, -16(%rbp)
	jmp	.L43
.L31:
	movq	$12, -16(%rbp)
	jmp	.L43
.L27:
	movl	$3, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L43
.L34:
	movl	-28(%rbp), %eax
	jmp	.L58
.L35:
	movl	$11, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L43
.L30:
	movl	$1, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L43
.L40:
	movl	$5, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L43
.L33:
	cmpl	$7, -24(%rbp)
	jle	.L53
	movq	$5, -16(%rbp)
	jmp	.L43
.L53:
	movq	$6, -16(%rbp)
	jmp	.L43
.L38:
	cmpl	$6, -24(%rbp)
	jne	.L55
	movq	$13, -16(%rbp)
	jmp	.L43
.L55:
	movq	$11, -16(%rbp)
	jmp	.L43
.L59:
	nop
.L43:
	jmp	.L57
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	scoreByWord, .-scoreByWord
	.section	.rodata
.LC0:
	.string	"%d"
	.align 8
.LC1:
	.string	"Usage: %s word1 word2 word3 ... (check if the word is a string)"
	.align 8
.LC2:
	.string	"Usage: %s word1 word2 word3 ..."
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_HkCk_envp(%rip)
	nop
.L61:
	movq	$0, _TIG_IZ_HkCk_argv(%rip)
	nop
.L62:
	movl	$0, _TIG_IZ_HkCk_argc(%rip)
	nop
	nop
.L63:
.L64:
#APP
# 113 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HkCk--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_HkCk_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_HkCk_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_HkCk_envp(%rip)
	nop
	movq	$14, -8(%rbp)
.L85:
	cmpq	$14, -8(%rbp)
	ja	.L87
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L67(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L67(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L67:
	.long	.L76-.L67
	.long	.L75-.L67
	.long	.L87-.L67
	.long	.L87-.L67
	.long	.L74-.L67
	.long	.L73-.L67
	.long	.L87-.L67
	.long	.L72-.L67
	.long	.L71-.L67
	.long	.L70-.L67
	.long	.L87-.L67
	.long	.L87-.L67
	.long	.L69-.L67
	.long	.L68-.L67
	.long	.L66-.L67
	.text
.L74:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -24(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L77
.L66:
	cmpl	$1, -36(%rbp)
	jg	.L78
	movq	$0, -8(%rbp)
	jmp	.L77
.L78:
	movq	$4, -8(%rbp)
	jmp	.L77
.L69:
	movl	-16(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jge	.L80
	movq	$9, -8(%rbp)
	jmp	.L77
.L80:
	movq	$13, -8(%rbp)
	jmp	.L77
.L71:
	cmpl	$0, -24(%rbp)
	je	.L82
	movq	$5, -8(%rbp)
	jmp	.L77
.L82:
	movq	$1, -8(%rbp)
	jmp	.L77
.L75:
	movl	$0, -20(%rbp)
	movl	$1, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L77
.L70:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	scoreByWord
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	addl	%eax, -20(%rbp)
	addl	$1, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L77
.L68:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L77
.L73:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$254, %edi
	call	exit@PLT
.L76:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$253, %edi
	call	exit@PLT
.L72:
	movl	$0, %eax
	jmp	.L86
.L87:
	nop
.L77:
	jmp	.L85
.L86:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
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
