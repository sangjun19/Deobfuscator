	.file	"TosakaUCW_COMP1023-FoC_L5T5T330016056_flatten.c"
	.text
	.globl	_TIG_IZ_1whQ_argv
	.bss
	.align 8
	.type	_TIG_IZ_1whQ_argv, @object
	.size	_TIG_IZ_1whQ_argv, 8
_TIG_IZ_1whQ_argv:
	.zero	8
	.globl	_TIG_IZ_1whQ_envp
	.align 8
	.type	_TIG_IZ_1whQ_envp, @object
	.size	_TIG_IZ_1whQ_envp, 8
_TIG_IZ_1whQ_envp:
	.zero	8
	.globl	_TIG_IZ_1whQ_argc
	.align 4
	.type	_TIG_IZ_1whQ_argc, @object
	.size	_TIG_IZ_1whQ_argc, 4
_TIG_IZ_1whQ_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter a letter grade: "
.LC1:
	.string	"%c"
.LC2:
	.string	"1.00"
.LC3:
	.string	"4.00"
.LC4:
	.string	"3.00"
.LC5:
	.string	"2.00"
.LC6:
	.string	"0.00"
.LC7:
	.string	"warning!"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1whQ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_1whQ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_1whQ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1whQ--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_1whQ_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_1whQ_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_1whQ_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L28:
	cmpq	$14, -16(%rbp)
	ja	.L31
	movq	-16(%rbp), %rax
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
	.long	.L17-.L8
	.long	.L31-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L31-.L8
	.long	.L14-.L8
	.long	.L31-.L8
	.long	.L13-.L8
	.long	.L31-.L8
	.long	.L12-.L8
	.long	.L31-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-17(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$13, -16(%rbp)
	jmp	.L18
.L10:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L18
.L15:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L18
.L11:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L18
.L12:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L18
.L9:
	movzbl	-17(%rbp), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	cmpl	$37, %eax
	ja	.L19
	movl	%eax, %eax
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
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L19-.L21
	.long	.L20-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L19-.L21
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L19-.L21
	.long	.L20-.L21
	.text
.L20:
	movq	$0, -16(%rbp)
	jmp	.L26
.L22:
	movq	$12, -16(%rbp)
	jmp	.L26
.L23:
	movq	$9, -16(%rbp)
	jmp	.L26
.L24:
	movq	$11, -16(%rbp)
	jmp	.L26
.L25:
	movq	$3, -16(%rbp)
	jmp	.L26
.L19:
	movq	$7, -16(%rbp)
	nop
.L26:
	jmp	.L18
.L14:
	movq	$14, -16(%rbp)
	jmp	.L18
.L17:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L18
.L13:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L18
.L16:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L30
.L31:
	nop
.L18:
	jmp	.L28
.L30:
	call	__stack_chk_fail@PLT
.L29:
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
