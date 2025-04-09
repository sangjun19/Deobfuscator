	.file	"kramlex_nsu-prog_8_flatten.c"
	.text
	.globl	_TIG_IZ_Z9rH_envp
	.bss
	.align 8
	.type	_TIG_IZ_Z9rH_envp, @object
	.size	_TIG_IZ_Z9rH_envp, 8
_TIG_IZ_Z9rH_envp:
	.zero	8
	.globl	_TIG_IZ_Z9rH_argv
	.align 8
	.type	_TIG_IZ_Z9rH_argv, @object
	.size	_TIG_IZ_Z9rH_argv, 8
_TIG_IZ_Z9rH_argv:
	.zero	8
	.globl	_TIG_IZ_Z9rH_argc
	.align 4
	.type	_TIG_IZ_Z9rH_argc, @object
	.size	_TIG_IZ_Z9rH_argc, 4
_TIG_IZ_Z9rH_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"main"
.LC1:
	.string	"kramlex_nsu-prog_8.c"
.LC2:
	.string	"argc == 2"
.LC3:
	.string	"%llu"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_Z9rH_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Z9rH_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Z9rH_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Z9rH--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_Z9rH_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_Z9rH_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_Z9rH_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L15:
	cmpq	$4, -16(%rbp)
	je	.L6
	cmpq	$4, -16(%rbp)
	ja	.L17
	cmpq	$3, -16(%rbp)
	je	.L8
	cmpq	$3, -16(%rbp)
	ja	.L17
	cmpq	$0, -16(%rbp)
	je	.L9
	cmpq	$1, -16(%rbp)
	je	.L10
	jmp	.L17
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rcx
	movl	$39, %edx
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L10:
	movq	-32(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	words_counter
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L11
.L8:
	cmpl	$2, -20(%rbp)
	jne	.L12
	movq	$1, -16(%rbp)
	jmp	.L11
.L12:
	movq	$4, -16(%rbp)
	jmp	.L11
.L9:
	movl	$0, %eax
	jmp	.L16
.L17:
	nop
.L11:
	jmp	.L15
.L16:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC4:
	.string	"r"
	.text
	.globl	words_counter
	.type	words_counter, @function
words_counter:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	$4, -16(%rbp)
.L50:
	cmpq	$20, -16(%rbp)
	ja	.L52
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L37-.L21
	.long	.L36-.L21
	.long	.L52-.L21
	.long	.L35-.L21
	.long	.L34-.L21
	.long	.L33-.L21
	.long	.L32-.L21
	.long	.L31-.L21
	.long	.L52-.L21
	.long	.L30-.L21
	.long	.L29-.L21
	.long	.L28-.L21
	.long	.L27-.L21
	.long	.L52-.L21
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L52-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L20-.L21
	.text
.L23:
	movb	$0, -49(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L38
.L34:
	movq	$3, -16(%rbp)
	jmp	.L38
.L26:
	cmpb	$-1, -50(%rbp)
	jne	.L39
	movq	$19, -16(%rbp)
	jmp	.L38
.L39:
	movq	$17, -16(%rbp)
	jmp	.L38
.L25:
	movl	$1, %edi
	call	exit@PLT
.L27:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$20, -16(%rbp)
	jmp	.L38
.L36:
	movq	$0, -32(%rbp)
	movb	$1, -49(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L38
.L35:
	movq	-72(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L38
.L28:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-50(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L41
	movq	$7, -16(%rbp)
	jmp	.L38
.L41:
	movq	$14, -16(%rbp)
	jmp	.L38
.L30:
	cmpq	$0, -40(%rbp)
	jne	.L43
	movq	$15, -16(%rbp)
	jmp	.L38
.L43:
	movq	$1, -16(%rbp)
	jmp	.L38
.L22:
	movb	$1, -49(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L38
.L24:
	cmpb	$0, -49(%rbp)
	je	.L45
	movq	$6, -16(%rbp)
	jmp	.L38
.L45:
	movq	$18, -16(%rbp)
	jmp	.L38
.L32:
	addq	$1, -32(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L38
.L33:
	cmpl	$0, -48(%rbp)
	je	.L47
	movq	$12, -16(%rbp)
	jmp	.L38
.L47:
	movq	$10, -16(%rbp)
	jmp	.L38
.L29:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movb	%al, -50(%rbp)
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L38
.L37:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -48(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L38
.L31:
	movb	$1, -49(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L38
.L20:
	movq	-32(%rbp), %rax
	jmp	.L51
.L52:
	nop
.L38:
	jmp	.L50
.L51:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	words_counter, .-words_counter
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
