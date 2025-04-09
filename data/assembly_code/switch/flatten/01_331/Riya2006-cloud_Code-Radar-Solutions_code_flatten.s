	.file	"Riya2006-cloud_Code-Radar-Solutions_code_flatten.c"
	.text
	.globl	_TIG_IZ_874V_envp
	.bss
	.align 8
	.type	_TIG_IZ_874V_envp, @object
	.size	_TIG_IZ_874V_envp, 8
_TIG_IZ_874V_envp:
	.zero	8
	.globl	_TIG_IZ_874V_argc
	.align 4
	.type	_TIG_IZ_874V_argc, @object
	.size	_TIG_IZ_874V_argc, 4
_TIG_IZ_874V_argc:
	.zero	4
	.globl	_TIG_IZ_874V_argv
	.align 8
	.type	_TIG_IZ_874V_argv, @object
	.size	_TIG_IZ_874V_argv, 8
_TIG_IZ_874V_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"Wednesday"
.LC2:
	.string	"Saturday"
.LC3:
	.string	"Thursday"
.LC4:
	.string	"Sunday"
.LC5:
	.string	"Monday"
.LC6:
	.string	"Tuesday"
.LC7:
	.string	"Invalid"
.LC8:
	.string	"Friday"
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
	movq	$0, _TIG_IZ_874V_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_874V_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_874V_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-874V--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_874V_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_874V_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_874V_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L31:
	cmpq	$17, -16(%rbp)
	ja	.L34
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
	.long	.L34-.L8
	.long	.L18-.L8
	.long	.L34-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L34-.L8
	.long	.L10-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L16:
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$10, -16(%rbp)
	jmp	.L19
.L18:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L17:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L32
	jmp	.L33
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L11:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L10:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L7:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L14:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L15:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L12:
	movl	-20(%rbp), %eax
	cmpl	$7, %eax
	ja	.L21
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L21-.L23
	.long	.L29-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L25-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L22:
	movq	$13, -16(%rbp)
	jmp	.L30
.L24:
	movq	$16, -16(%rbp)
	jmp	.L30
.L25:
	movq	$7, -16(%rbp)
	jmp	.L30
.L26:
	movq	$11, -16(%rbp)
	jmp	.L30
.L27:
	movq	$1, -16(%rbp)
	jmp	.L30
.L28:
	movq	$6, -16(%rbp)
	jmp	.L30
.L29:
	movq	$17, -16(%rbp)
	jmp	.L30
.L21:
	movq	$5, -16(%rbp)
	nop
.L30:
	jmp	.L19
.L13:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L19
.L34:
	nop
.L19:
	jmp	.L31
.L33:
	call	__stack_chk_fail@PLT
.L32:
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
