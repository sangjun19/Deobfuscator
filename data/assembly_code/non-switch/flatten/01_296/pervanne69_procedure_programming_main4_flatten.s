	.file	"pervanne69_procedure_programming_main4_flatten.c"
	.text
	.globl	_TIG_IZ_rs3E_envp
	.bss
	.align 8
	.type	_TIG_IZ_rs3E_envp, @object
	.size	_TIG_IZ_rs3E_envp, 8
_TIG_IZ_rs3E_envp:
	.zero	8
	.globl	_TIG_IZ_rs3E_argc
	.align 4
	.type	_TIG_IZ_rs3E_argc, @object
	.size	_TIG_IZ_rs3E_argc, 4
_TIG_IZ_rs3E_argc:
	.zero	4
	.globl	_TIG_IZ_rs3E_argv
	.align 8
	.type	_TIG_IZ_rs3E_argv, @object
	.size	_TIG_IZ_rs3E_argv, 8
_TIG_IZ_rs3E_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\n"
.LC1:
	.string	"Address of A = %p\n"
.LC2:
	.string	"Address of B = %p\n\n"
.LC3:
	.string	"A: \n"
.LC4:
	.string	"%d "
.LC5:
	.string	"\n\nB\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_rs3E_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_rs3E_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_rs3E_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 154 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-rs3E--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_rs3E_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_rs3E_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_rs3E_envp(%rip)
	nop
	movq	$12, -48(%rbp)
.L33:
	cmpq	$28, -48(%rbp)
	ja	.L36
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L36-.L8
	.long	.L20-.L8
	.long	.L36-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L36-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L15-.L8
	.long	.L36-.L8
	.long	.L14-.L8
	.long	.L36-.L8
	.long	.L13-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L12-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L11-.L8
	.long	.L36-.L8
	.long	.L10-.L8
	.long	.L36-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-64(%rbp), %rax
	movl	$200, %esi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	-56(%rbp), %rax
	movl	$800, %esi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -76(%rbp)
	movq	$28, -48(%rbp)
	jmp	.L23
.L14:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$27, -48(%rbp)
	jmp	.L23
.L15:
	movq	$1, -48(%rbp)
	jmp	.L23
.L17:
	cmpl	$199, -76(%rbp)
	jg	.L24
	movq	$6, -48(%rbp)
	jmp	.L23
.L24:
	movq	$14, -48(%rbp)
	jmp	.L23
.L21:
	movl	$100, -72(%rbp)
	movl	$100, -68(%rbp)
	movl	-72(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	-68(%rbp), %eax
	cltq
	movl	$4, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -76(%rbp)
	movq	$9, -48(%rbp)
	jmp	.L23
.L11:
	movq	-64(%rbp), %rdx
	movl	-76(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -76(%rbp)
	movq	$9, -48(%rbp)
	jmp	.L23
.L20:
	movq	-64(%rbp), %rdx
	movl	-76(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -76(%rbp)
	movq	$28, -48(%rbp)
	jmp	.L23
.L13:
	movl	-76(%rbp), %eax
	cmpl	-68(%rbp), %eax
	jge	.L26
	movq	$0, -48(%rbp)
	jmp	.L23
.L26:
	movq	$25, -48(%rbp)
	jmp	.L23
.L16:
	movl	-76(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jge	.L28
	movq	$23, -48(%rbp)
	jmp	.L23
.L28:
	movq	$5, -48(%rbp)
	jmp	.L23
.L12:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -76(%rbp)
	movq	$8, -48(%rbp)
	jmp	.L23
.L18:
	movq	-56(%rbp), %rdx
	movl	-76(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -76(%rbp)
	movq	$8, -48(%rbp)
	jmp	.L23
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L7:
	cmpl	$49, -76(%rbp)
	jg	.L31
	movq	$3, -48(%rbp)
	jmp	.L23
.L31:
	movq	$19, -48(%rbp)
	jmp	.L23
.L19:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -76(%rbp)
	movq	$16, -48(%rbp)
	jmp	.L23
.L22:
	movq	-56(%rbp), %rdx
	movl	-76(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -76(%rbp)
	movq	$16, -48(%rbp)
	jmp	.L23
.L36:
	nop
.L23:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
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
