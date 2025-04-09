	.file	"HAliveKP_c-code_e_flatten.c"
	.text
	.globl	_TIG_IZ_oCgK_argv
	.bss
	.align 8
	.type	_TIG_IZ_oCgK_argv, @object
	.size	_TIG_IZ_oCgK_argv, 8
_TIG_IZ_oCgK_argv:
	.zero	8
	.globl	_TIG_IZ_oCgK_argc
	.align 4
	.type	_TIG_IZ_oCgK_argc, @object
	.size	_TIG_IZ_oCgK_argc, 4
_TIG_IZ_oCgK_argc:
	.zero	4
	.globl	_TIG_IZ_oCgK_envp
	.align 8
	.type	_TIG_IZ_oCgK_envp, @object
	.size	_TIG_IZ_oCgK_envp, 8
_TIG_IZ_oCgK_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d "
.LC1:
	.string	"Elements of the array:"
.LC2:
	.string	"Enter 10 elements:"
.LC3:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_oCgK_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_oCgK_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_oCgK_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 90 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-oCgK--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_oCgK_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_oCgK_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_oCgK_envp(%rip)
	nop
	movq	$8, -56(%rbp)
.L23:
	cmpq	$15, -56(%rbp)
	ja	.L26
	movq	-56(%rbp), %rax
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
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L26-.L8
	.long	.L27-.L8
	.long	.L10-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	-60(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -60(%rbp)
	movq	$3, -56(%rbp)
	jmp	.L17
.L9:
	cmpl	$9, -64(%rbp)
	jg	.L18
	movq	$2, -56(%rbp)
	jmp	.L17
.L18:
	movq	$11, -56(%rbp)
	jmp	.L17
.L7:
	movl	$10, %edi
	call	putchar@PLT
	movq	$10, -56(%rbp)
	jmp	.L17
.L12:
	movq	$7, -56(%rbp)
	jmp	.L17
.L15:
	cmpl	$9, -60(%rbp)
	jg	.L20
	movq	$4, -56(%rbp)
	jmp	.L17
.L20:
	movq	$15, -56(%rbp)
	jmp	.L17
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -60(%rbp)
	movq	$3, -56(%rbp)
	jmp	.L17
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -64(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L17
.L16:
	movl	-64(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -64(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L17
.L26:
	nop
.L17:
	jmp	.L23
.L27:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L25
	call	__stack_chk_fail@PLT
.L25:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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
