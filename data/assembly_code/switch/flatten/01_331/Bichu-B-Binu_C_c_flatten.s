	.file	"Bichu-B-Binu_C_c_flatten.c"
	.text
	.globl	_TIG_IZ_X8OK_argc
	.bss
	.align 4
	.type	_TIG_IZ_X8OK_argc, @object
	.size	_TIG_IZ_X8OK_argc, 4
_TIG_IZ_X8OK_argc:
	.zero	4
	.globl	_TIG_IZ_X8OK_envp
	.align 8
	.type	_TIG_IZ_X8OK_envp, @object
	.size	_TIG_IZ_X8OK_envp, 8
_TIG_IZ_X8OK_envp:
	.zero	8
	.globl	_TIG_IZ_X8OK_argv
	.align 8
	.type	_TIG_IZ_X8OK_argv, @object
	.size	_TIG_IZ_X8OK_argv, 8
_TIG_IZ_X8OK_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d\t"
.LC1:
	.string	"%d"
.LC2:
	.string	"Entered number is:"
.LC3:
	.string	"Enter the metrix numbers:"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movq	$0, _TIG_IZ_X8OK_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_X8OK_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_X8OK_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-X8OK--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_X8OK_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_X8OK_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_X8OK_envp(%rip)
	nop
	movq	$0, -56(%rbp)
.L32:
	cmpq	$23, -56(%rbp)
	ja	.L35
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L35-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L35-.L8
	.long	.L13-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L12-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L9-.L8
	.long	.L35-.L8
	.long	.L7-.L8
	.text
.L17:
	movl	-60(%rbp), %eax
	movslq	%eax, %rcx
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rcx, %rax
	movl	-48(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -60(%rbp)
	movq	$23, -56(%rbp)
	jmp	.L22
.L14:
	movl	$0, -60(%rbp)
	movq	$16, -56(%rbp)
	jmp	.L22
.L20:
	cmpl	$2, -64(%rbp)
	jg	.L23
	movq	$21, -56(%rbp)
	jmp	.L22
.L23:
	movq	$13, -56(%rbp)
	jmp	.L22
.L7:
	cmpl	$2, -60(%rbp)
	jg	.L25
	movq	$4, -56(%rbp)
	jmp	.L22
.L25:
	movq	$17, -56(%rbp)
	jmp	.L22
.L18:
	addl	$1, -64(%rbp)
	movq	$2, -56(%rbp)
	jmp	.L22
.L11:
	cmpl	$2, -60(%rbp)
	jg	.L27
	movq	$5, -56(%rbp)
	jmp	.L22
.L27:
	movq	$3, -56(%rbp)
	jmp	.L22
.L9:
	movl	$0, -60(%rbp)
	movq	$23, -56(%rbp)
	jmp	.L22
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L10:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -64(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L22
.L16:
	leaq	-48(%rbp), %rcx
	movl	-60(%rbp), %eax
	movslq	%eax, %rsi
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -60(%rbp)
	movq	$16, -56(%rbp)
	jmp	.L22
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -64(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L22
.L21:
	movq	$7, -56(%rbp)
	jmp	.L22
.L15:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -64(%rbp)
	movq	$2, -56(%rbp)
	jmp	.L22
.L19:
	cmpl	$2, -64(%rbp)
	jg	.L30
	movq	$8, -56(%rbp)
	jmp	.L22
.L30:
	movq	$10, -56(%rbp)
	jmp	.L22
.L35:
	nop
.L22:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
